import json
import random
import os
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from abc import ABC
from dotenv import load_dotenv


import openai
import anthropic

load_dotenv()

TOKEN_LOG_FILE = "token_usage.txt"

#IMPORTANT COMMENTS: 
#if you change anything in the game settings (rules, market dynamics ...) be sure that this change is represented in the game-intro for agents. Check initialize_agent method in the class Company(ABC)

#README вертикально интегрируемая компания 
#FIXME make backlog fee as a contract parameter
#


"""Notes 
ChatGPT: 
Backlog fees are large relative to typical per-unit margins (e.g., a Factory fixed sale yields ~0.2 premium over base 2.0, but backlog penalties are around 2.0 per undelivered unit each round). This makes over-promising extremely unattractive, encouraging conservative commitments, flexible contracts, or mutual cancellations in tight markets.
"""

# Enums for contract types and company types
class ContractType(Enum):
    FIXED = "fixed"
    FLEXIBLE = "flexible"

class CompanyType(Enum):
    FACTORY = "factory"
    WHOLESALE = "wholesale"
    RETAIL = "retail"

# Global Parameters Configuration
@dataclass
class GlobalParameters:
    num_rounds: int = 2
    max_messages_per_call: int = 10
    max_factory_production: int = 500
    starting_money: float = 10000.0
    starting_beer: int = 300
    default_max_storage: int = 1000
    storage_cost_per_unit: float = 0.5
    base_prices: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0])
    contract_multipliers: Dict[str, float] = field(default_factory=lambda: {"fixed": 1.1, "flexible": 1.2})
    # flexible contracts has to have a higher number than fixed, as their conditions profit buyer. In case it lower you have to change order of proceeding them in case of goods shortage in handle_contracts_and_orders 
    flexibility_percentage: float = 0.2
    #backlog_fee: float = 2.0 currently equal to the base price
    contracts_enabled: bool = True
    production_cost: float = 0.0

# Contract Class
@dataclass
class Contract:
    #N.B. if you change contract constructor don't forget to change the system prompt as agents may use wrong format otherwise
    contract_id: int
    parties: Tuple[str, str]  # (supplier, buyer)
    start_round: int
    length: int
    amount: int
    contract_type: ContractType
    fine: float
    price_per_unit: float
    
    def is_active(self, current_round: int) -> bool:
        return self.start_round <= current_round < self.start_round + self.length
    
    def get_flexible_range(self, flexibility: float) -> Tuple[int, int]:
        if self.contract_type == ContractType.FLEXIBLE:
            min_amount = int(self.amount * (1 - flexibility))
            max_amount = int(self.amount * (1 + flexibility))
            return min_amount, max_amount
        return self.amount, self.amount

# Database for logging
class Database:
    def __init__(self):
        self.transactions = []
        self.dialogues = []
        self.round_states = []
        self.contracts_log = []
        self.market_demand_log = []
        self.models_used = {}

    def log_models_used(self, models: Dict[str, Dict[str, Any]]):
        self.models_used = models

    def log_transaction(self, round_num: int, from_company: str, to_company: str, 
                       beer_amount: int, money_amount: float, transaction_type: str):
        self.transactions.append({
            "round": round_num,
            "from": from_company,
            "to": to_company,
            "beer": beer_amount,
            "money": money_amount,
            "type": transaction_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_dialogue(self, round_num: int, company1: str, company2: str, messages: List[Dict]):
        self.dialogues.append({
            "round": round_num,
            "participants": [company1, company2],
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_round_state(self, round_num: int, company_states: Dict):
        self.round_states.append({
            "round": round_num,
            "states": company_states,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_contract(self, contract: Contract, action: str):
        # Convert contract to dict, handling the enum
        contract_dict = asdict(contract)
        # Convert ContractType enum to string
        contract_dict['contract_type'] = contract.contract_type.value
    
        self.contracts_log.append({
            "contract": contract_dict,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_market_demand(self, round_num: int, demand: int):
        self.market_demand_log.append({
            "round": round_num,
            "demand": demand,
            "timestamp": datetime.now().isoformat()
        })
    
    def export_to_json(self, filename: str):
        data = {
            "transactions": self.transactions,
            "dialogues": self.dialogues,
            "round_states": self.round_states,
            "contracts": self.contracts_log,
            "market_demand": self.market_demand_log,
            "models_used": self.models_used
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Base Company Class
class Company(ABC):
    def __init__(self, name: str, company_type: CompanyType, params: GlobalParameters, llm_interface):
        self.name = name
        self.company_type = company_type
        self.money = params.starting_money
        self.beer_storage = params.starting_beer
        self.max_storage = params.default_max_storage
        self.backlog = {}  # {company_name: amount}
        self.contracts = []
        self.params = params
        self.llm = llm_interface  # This would be the actual LLM interface
        self.message_buffer = []

    def initialize_agent(self):
        """Send initial game rules and company-specific information to the agent"""

        introduction = f"""You are participating in a a modified beer supply chain game.
        
        SUPPLY CHAIN STRUCTURE:
        Factory → Wholesale → Retail → Market

        First three are operated by AI agents and you are one of them, you operate the {self.name} company. Your goal is to maximize long term profit of your company and make it stable over the period (minimize round-by-round profit fluctuations).

        """
        
        self.add_message_to_buffer(introduction)

        market_info = """
        Market represents demand from customers. You know from your past experience that is quite stable with very rare disturbances. Only retail knows demand for the current round. 

        """

        #this prompt is very important for realistic simulation
        #if a company introduces an AI agent for operational goal
        #they will 100% provide all the useful information to it including known information about the market

        self.add_message_to_buffer(market_info)

        rules = f"""
        KEY MECHANICS:
        
        1. ORDERING: Each round, Wholesale and Retail can order beer from their suppliers
        - Orders are delivered NEXT round (1-round delay)
        - Payment happens instantly
        Factory produces beer each round based on decision made by the operator during the previous round. 
       
        2. CONVERSATIONS: You can talk and negotiate with neighbors
        - Keep messages concise
        - The maximum messages you can send to each other during single conversation is {self.params.max_messages_per_call}
        - You can't sign contracts or orders during one-to-one conversations (see below). You can discuss any details but official signing is a different procedure in the next phase of the game.
        - Write "TERMINATE_CONVERSATION" to end conversation early.
        
        3. CONTRACTS: You can create long-term contracts
        - Fixed contracts: Guaranteed amount each round at {int((self.params.contract_multipliers['fixed']-1)*100)}% premium
        - Flexible contracts: ±{int(self.params.flexibility_percentage*100)}% variable amount at {int((self.params.contract_multipliers['flexible']-1)*100+1)}% premium. Amount is determined by the buyer.
        - Contracts can be broken by paying the fine or canceled by mutual agreement of both parties in case of supply shortage.
        - The formal negotiation has only two turns and happens after conversations. 1. The buyer proposes the contract. 2. The seller may agree, reject, or propose another one. 3. If the seller proposes a new one the buyer can only agree or reject.
        - Buyer may order beer for the next round for the default price (see below) without signing contracts. This can be done right after negotiation of contracts. One-time orders are ALWAYS additional to existing obligations; never treat them as substitutes."
        - Between two companies only a single contract per round can be signed. 
        
        4. SHORTAGES: If supplier can't fulfill orders:
        - Contract review sequence: Flexible contracts → Fixed contracts → One-time orders/Backlog
        - Unfulfilled shipments become backlog
        - Backlog fee charged each round until fulfilled
        
        5. COSTS:
        - Storage costs charged each round for all inventory
        - Excess beer above max storage is destroyed
        - Negative money of any company ends the game with a loss for everyone
        
        PRICES:
        - Factory→Wholesale: ${self.params.base_prices[0]}/unit base
        - Wholesale→Retail: ${self.params.base_prices[1]}//unit base  
        - Retail→Market: ${self.params.base_prices[2]}//unit base

        FORMAT REQUIREMENTS:
        - Yes/No questions: Answer only "Yes" or "No"
        - Numbers: Provide only the number
        - Contracts: type,amount,length,fine (e.g., "fixed,100,10,500")

        """

        self.add_message_to_buffer(rules)

        starting_conditions = f"""
        Your Starting Conditions:
        - Money: ${self.money}
        - Beer in storage: {self.beer_storage} units
        - Maximum storage: {self.max_storage} units
        
        Key Economic Parameters:
        - Storage cost: ${self.params.storage_cost_per_unit} per unit per round
        - Backlog fee: beer price per unit per round
        - Contract multipliers: Fixed={self.params.contract_multipliers['fixed']}x, Flexible={self.params.contract_multipliers['flexible']}x
        - Flexible contract range: ±{self.params.flexibility_percentage * 100}%
        
        {"Production capacity: " + str(self.max_production) + " units/round" if hasattr(self, 'max_production') else ""}
        
        Remember: All deliveries have a 1-round delay. Beer ordered in Round N arrives in Round N+1.
        
        """
        
        self.add_message_to_buffer(starting_conditions)
        
    def get_state(self, current_round: int) -> Dict:
        return {
            "name": self.name,
            "type": self.company_type.value,
            "money": self.money,
            "beer_storage": self.beer_storage,
            "backlog": self.backlog,
            "active_contracts": len([c for c in self.contracts if c.is_active(current_round)])
        }
    
    def add_message_to_buffer(self, message: str):
        """Add message to buffer without sending to LLM"""
        self.message_buffer.append(message)
    
    def get_llm_response(self, prompt: str, format_instruction: str = "") -> str:
        """Send accumulated messages plus prompt to LLM and get response"""
        full_prompt = "\n".join(self.message_buffer) + "\n" + prompt
        if format_instruction:
            full_prompt += f"\n{format_instruction}"
        self.message_buffer = []  # Clear buffer after sending
        
        return self.llm.generate(full_prompt) #if self.llm else "Yes"
    
    def propose_contract(self, other_company: 'Company', contract_details: Dict) -> Optional[Contract]:
        """Propose a contract to another company"""
        prompt = f"""
        You have been offered a contract by {self.name}:
        Type: {contract_details['type']}
        Amount per round: {contract_details['amount']}
        Duration: {contract_details['length']} rounds
        Fine for breaking: {contract_details['fine']}
        Price per unit: {contract_details['price_per_unit']}

        \nNOTE: New contracts start THIS round (obligation counts now). 
        Deliveries you send THIS round arrive to the counterparty NEXT round.
        
        Do you accept this contract?
        """
        
        response = other_company.get_llm_response(prompt, "Answer with only one word: Accept, Reject, or Counter (for a counter-offer)")
        
        if "accept" in response.lower():
            contract = game.create_contract(other_company, self, contract_details)
            return contract
        elif "counter" in response.lower():
            # Get counter proposal from other company
            counter_prompt = "Propose your counter-offer"
            counter = other_company.get_llm_response(
                counter_prompt, 
                "Format: type,amount,length,fine (Example: flexible,150,10,500)"
            )
            
            # Parse counter-offer
            try:
                parts = counter.split(',')
                if len(parts) == 4:
                    counter_type = parts[0].strip().lower()
                    counter_amount = int(parts[1].strip())
                    counter_length = int(parts[2].strip())
                    counter_fine = float(parts[3].strip())
                    
                    # Calculate price based on contract type
                    base_price_index = 0 if self.company_type == CompanyType.FACTORY else 1
                    base_price = self.params.base_prices[base_price_index]
                    
                    if counter_type == "flexible":
                        counter_price = base_price * self.params.contract_multipliers["flexible"]
                    elif counter_type == "fixed":
                        counter_price = base_price * self.params.contract_multipliers["fixed"]
                    else:
                        print("\n⚠️  WARNING: uknown contract type: ⚠️" + counter_type +"\n")
                        return None  # Invalid type
                    
                    # Ask original proposer to accept counter-offer
                    accept_prompt = f"""
                    Counter-offer received:
                    Type: {counter_type}
                    Amount per round: {counter_amount}
                    Duration: {counter_length} rounds
                    Fine for breaking: {counter_fine}
                    Price per unit: {counter_price:.2f} (auto-calculated)
                    
                    Do you accept? Answer: Yes/No
                    """
                    
                    accept_response = self.get_llm_response(accept_prompt, "Answer only: Yes or No")
                    
                    if "yes" in accept_response.lower():

                        counter_details = {
                            "type": counter_type,
                            "amount": counter_amount,
                            "length": counter_length,
                            "fine": counter_fine,
                            "price_per_unit": counter_price,
                            "start_round": contract_details["start_round"],
                        }

                        contract = game.create_contract(other_company, self, counter_details)

                        self.add_message_to_buffer("The contract has been signed\n")
                        other_company.add_message_to_buffer("The contract has been signed\n")

                        return contract
                    else:
                        other_company.add_message_to_buffer("Your proposal was rejected\n")
                        self.add_message_to_buffer("you rejected the contract.\n")

            except (ValueError, IndexError):
                # Failed to parse counter-offer
                print("\n⚠️  WARNING: Problem with parsing ⚠️")
                print("It has to be a counter-offer. Format: type,amount,length,fine (Example: flexible,150,10,500). But: \n")
                print(counter)
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    self.game_active = False
                    return 0
                pass
        
            return None
        else:
            self.add_message_to_buffer("You rejected the contract.\n\n")
            other_company.add_message_to_buffer("Your proposal was rejected.\n\n")
            return None
    
    def call_neighbor(self, other_company: 'Company', max_messages: int) -> List[Dict]:
        """Have a conversation with another company"""
        conversation = []
        
        initial_prompt = f"""
        I'm opening a chat with {other_company.name}. 
        Discuss ONLY contract terms (type, amount, length,fine). 
        Price is predefined by the game and is calculated authomaticaly, do NOT discuss it.
        Do NOT discuss product types, pallets, SKUs, tracking numbers, or logistics. 
        One short sentence max. Write 'TERMINATE_CONVERSATION' to end.
        
        Write the first message.\n"""
        message = self.get_llm_response(
            initial_prompt
        )
        
        for i in range(max_messages):
            conversation.append({"from": self.name, "message": message})
            response_prompt = None
            
            if i==0: 
                response_prompt = f"""
                I'm oppening a chat with {self.name}...

                Discuss ONLY contract terms (type, amount, length,fine). 
                Price is predefined by the game and is calculated authomaticaly, do NOT discuss it.
                Do NOT discuss product types, pallets, SKUs, tracking numbers, or logistics. 
                One short sentence max. Write 'TERMINATE_CONVERSATION' to end.


                {self.name} says: {message}\nWrite your response below."""
                
                response = other_company.get_llm_response(
                response_prompt, ""
                )
            else:            
                response_prompt = f"{self.name} says: {message}\nWrite your response below."
                response = other_company.get_llm_response(
                response_prompt, 
                """ Write 'TERMINATE_CONVERSATION' to disconect."""
                )
            conversation.append({"from": other_company.name, "message": response})
            
            # Check if responder wants to end conversation
            if "TERMINATE_CONVERSATION" in response.upper():
                self.add_message_to_buffer(response)
                self.add_message_to_buffer("\n The conversation has been ended.\n")
                break
            
            # Only continue if not at message limit and conversation hasn't ended
            if i < max_messages - 1:
                continue_prompt = f"{other_company.name} says: {response}\nWrite your response below. "
                message = self.get_llm_response(
                    continue_prompt, 
                    "Write 'TERMINATE_CONVERSATION' to disconect."
                )
                
                # Check if initiator wants to end conversation
                if "TERMINATE_CONVERSATION" in message.upper():
                    other_company.add_message_to_buffer(message)
                    other_company.add_message_to_buffer("\n The conversation has been ended.\n")
                    conversation.append({"from": self.name, "message": message})
                    break
        
        return conversation
    
    def pay_storage_costs(self) -> float:
        """Calculate and pay storage costs"""
        cost = self.beer_storage * self.params.storage_cost_per_unit
        prompt = f"""
        Storage cost at the end of the round: {cost}
        """
        self.add_message_to_buffer(prompt)
        self.money -= cost
        return cost
    
    def remove_excess_storage(self) -> int:
        """Remove beer exceeding max storage"""
        excess = max(0, self.beer_storage - self.max_storage)
        if excess > 0:
            self.beer_storage = self.max_storage
            self.add_message_to_buffer(f"Warning: {excess} units of beer removed due to storage limit")
        return excess
    

    def track_round_start(self):
        """Call at the beginning of each round to track starting values"""
        self.round_start_money = self.money
        self.round_start_beer = self.beer_storage
        self.round_start_backlog = sum(self.backlog.values())
    

    def calculate_round_metrics(self, round_num: int) -> Dict:
        """Calculate performance metrics for this specific round"""
        
        # Calculate changes for THIS round only
        money_change = self.money - self.round_start_money
        inventory_change = self.beer_storage - self.round_start_beer
        backlog_change = sum(self.backlog.values()) - self.round_start_backlog
        
        # Calculate costs incurred this round
        storage_cost = self.beer_storage * self.params.storage_cost_per_unit
        
        # Backlog fees paid this round (approximation based on backlog at round start)
        backlog_fee_per_unit = self.get_backlog_fee_per_unit()
        backlog_fees_paid = self.round_start_backlog * backlog_fee_per_unit

        
        return {
            'money_change': money_change,
            'inventory_change': inventory_change,
            'backlog_change': backlog_change,
            'storage_cost': storage_cost,
            'backlog_fees_paid': backlog_fees_paid,
            'ending_money': self.money,
            'ending_inventory': self.beer_storage,
            'ending_backlog': sum(self.backlog.values())
        }
    
    def get_backlog_fee_per_unit(self) -> float:
        """Get the backlog fee per unit based on company type"""
        if self.company_type == CompanyType.FACTORY:
            return self.params.base_prices[0]
        elif self.company_type == CompanyType.WHOLESALE:
            return self.params.base_prices[1]
        else:  # RETAIL
            return 0  # Retail doesn't pay backlog to market
        
    """    
    def get_total_obligations(self, round_num: int) -> int:
        #Calculate total delivery obligations for this round
        if self.company_type == CompanyType.RETAIL:
            return market_demand_function(round_num)
            
        # Contracts where we are the supplier
        contract_obligations = sum(
            c.amount for c in self.contracts 
            if c.parties[0] == self.name and c.is_active(round_num)
        )
        # Plus any backlog we owe
        backlog_obligations = sum(self.backlog.values())
        
        return contract_obligations + backlog_obligations
    
    """
    def provide_round_feedback(self, metrics: Dict, round_num: int):
        """Provide learning feedback to the agent after each round"""
        
        feedback = None

        if self.company_type == CompanyType.RETAIL:
            feedback = f"""
            === ROUND {round_num} PERFORMANCE REVIEW ===
            
            This Round's Changes:
            - Money change: ${metrics['money_change']:+.2f}
            - Inventory change: {metrics['inventory_change']:+d} units  
            
            Round Costs:
            - Storage cost: ${metrics['storage_cost']:.2f}
            """
        else:
            feedback = f"""
            === ROUND {round_num} PERFORMANCE REVIEW ===
            
            This Round's Changes:
            - Money change: ${metrics['money_change']:+.2f}
            - Inventory change: {metrics['inventory_change']:+d} units  
            - Backlog change: {metrics['backlog_change']:+d} units
            
            Round Costs:
            - Storage cost: ${metrics['storage_cost']:.2f}
            - Backlog penalties: ${metrics['backlog_fees_paid']:.2f}
            """
        
        self.add_message_to_buffer(feedback)


    def get_formatted_response(self, prompt: str, format_type: str, 
                          format_instruction: str, max_retries: int = 1) -> str:
        """
        Get a properly formatted response from LLM with retry logic
        
        Args:
            prompt: The main prompt
            format_type: Type of format expected ('number', 'yes_no', 'contract', etc.)
            format_instruction: Specific format instruction
            max_retries: Number of retries if format is wrong

            - yes_no - Simple yes/no questions
            - yes_no_number - "Yes", "No", OR a counter-offer number (one-time orders)
            - option_123 - Choose from 3 options (shortage handling)
            - option_12 - Choose from 2 options (after mutual cancellation refused)
            - number - Just extract a number
            - contract - The specific 4-part contract format
            - accept_reject_counter - For contract responses
            - raw - Any non-empty response
        
        Returns:
            Cleaned, validated response or None if failed
        """
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                # On retry, add error message
                error_prompt = f"""
                ERROR: Your previous answer was in the wrong format.
                You answered: {response}

                REQUIRED FORMAT: {format_instruction}
                Answer ONLY in the required format with nothing else.
                """
                response = self.get_llm_response(error_prompt, "")
            else:
                # First attempt
                response = self.get_llm_response(prompt, format_instruction)
            
            # Validate based on format type
            if format_type == "number":
                # Extract first number found
                numbers = re.findall(r'\d+', response)
                if numbers:
                    if len(numbers) > 1:
                        text = f"Several numbers were found in your responce {numbers}. The first was accepted.\n"
                        self.add_message_to_buffer(text)
                    return numbers[0]
                    
            elif format_type == "yes_no":
                response_lower = response.lower()
                if "yes" in response_lower and "no" not in response_lower:
                    return "yes"
                elif "no" in response_lower and "yes" not in response_lower:
                    return "no"
                                            
            elif format_type == "yes_no_number":
                # For responses that can be "yes", "no", or a number
                response_lower = response.lower()
                
                # Check for yes/no first (more specific)
                if "yes" in response_lower and "no" not in response_lower:
                    # Make sure it's not part of a number explanation
                    if not re.search(r'\d+', response):
                        return "yes"
                elif "no" in response_lower and "yes" not in response_lower:
                    if not re.search(r'\d+', response):
                        return "no"   
                elif re.search(r'\d+', response) and "yes" not in response_lower and "no" not in response_lower:
                    numbers = re.findall(r'\d+', response)
                    if numbers:
                        if len(numbers) > 1:
                            text = f"Several numbers were found in your responce {numbers}. The first was accepted.\n"
                            self.add_message_to_buffer(text)
                        return numbers[0]
                        
            elif format_type == "contract":
                # Check for exactly 4 comma-separated values
                lines = response.strip().split('\n')
                if len(lines) == 1:  # Only one line
                    parts = lines[0].split(',')
                    if len(parts) == 4:
                        # Validate contract type
                        contract_type = parts[0].strip().lower()
                        if contract_type in ["fixed", "flexible"]:
                            try:
                                # Validate numeric parts
                                int(parts[1].strip())  # amount
                                int(parts[2].strip())  # length
                                float(parts[3].strip())  # fine
                                return lines[0].strip()
                            except ValueError:
                                pass
                    
            elif format_type == "option_123":
                # For responses that should be 1, 2, or 3
                import re
                # Look for standalone 1, 2, or 3
                if "1" in response and "2" not in response and "3" not in response:
                    return "1"
                elif "2" in response and "1" not in response and "3" not in response:
                    return "2"
                elif "3" in response and "1" not in response and "2" not in response:
                    return "3"
                # Also check for written forms
                response_lower = response.lower()
                if "one" in response_lower or "first" in response_lower:
                    return "1"
                elif "two" in response_lower or "second" in response_lower:
                    return "2"
                elif "three" in response_lower or "third" in response_lower:
                    return "3"
                    
            elif format_type == "option_12":
                # For responses that should be 1 or 2
                if "1" in response and "2" not in response:
                    return "1"
                elif "2" in response and "1" not in response:
                    return "2"
                # Also check for written forms
                response_lower = response.lower()
                if "one" in response_lower or "first" in response_lower:
                    return "1"
                elif "two" in response_lower or "second" in response_lower:
                    return "2"
                    
            elif format_type == "accept_reject_counter":
                response_lower = response.lower()
                if "accept" in response_lower and "reject" not in response_lower and "counter" not in response_lower:
                    return "accept"
                elif "reject" in response_lower and "accept" not in response_lower and "counter" not in response_lower:
                    return "reject"
                elif "counter" in response_lower:
                    return "counter"
                    
            elif format_type == "raw":
                # For responses where we just need something
                if response.strip():
                    return response.strip()
        
        # Failed all attempts
        return None

# Factory Class
class Factory(Company):
    def __init__(self, name: str, params: GlobalParameters, llm_interface):
        super().__init__(name, CompanyType.FACTORY, params, llm_interface)
        self.max_production = params.max_factory_production
        self.production_cost = params.production_cost
        self.planned_production = 0
    
    def decide_production(self, round_num: int) -> int:
        """Decide how much to produce for next round"""
        info = f"""
        Current round: {round_num}
        Current storage: {self.beer_storage}
        Max storage: {self.max_storage}
        Max production: {self.max_production}
        Active contracts: {sum(c.amount for c in self.contracts if c.is_active(round_num))}
        Money: {self.money}
        Production cost per unit: {self.production_cost}
        """
        
        self.add_message_to_buffer(info)
        prompt = "How much beer should we produce for next round?"
        response = self.get_llm_response(prompt, f"Answer ONLY with a number between 0 and {self.max_production}")
        
        try:
            production = int(response)
            if production > self.max_production: 
                prompt = f"""
                You ordered {production} which is more than factory's max production {self.max_production}
                The factory will produce its maximum.
                """
                self.add_message_to_buffer(prompt)
            production = max(0, min(production, self.max_production))
        except:

            # Failed to read amount
            print("\n⚠️  WARNING: Problem with reading production amount (decide production) ⚠️")
            print("It has to be a number")
            print(response)
            response1 = input("Continue? (y/n): ")
            if response1.lower() != 'y':                    
                self.game_active = False
                return 0

            production = self.max_production // 2  # Default fallback
        
        self.planned_production = production
        self.money -= production * self.production_cost
    
    def produce_beer(self):
        """Execute planned production"""
        self.beer_storage += self.planned_production
        self.planned_production = 0

# Wholesale Class  
class Wholesale(Company):
    def __init__(self, name: str, params: GlobalParameters, llm_interface):
        super().__init__(name, CompanyType.WHOLESALE, params, llm_interface)

# Retail Class
class Retail(Company):
    def __init__(self, name: str, params: GlobalParameters, llm_interface):
        super().__init__(name, CompanyType.RETAIL, params, llm_interface)
        sale_amount = 0
    
    def decide_market_sale(self, market_demand: int) -> int:
        """Decide how much to sell to market"""
        info = f"""
        Market demand: {market_demand}
        Current storage: {self.beer_storage}
        Selling price: {self.params.base_prices[2]}
        """
        
        self.add_message_to_buffer(info)
        prompt = "How much beer should we sell to the market?"
        response = self.get_llm_response(prompt, f"Answer only a number between 0 and {min(market_demand, self.beer_storage)}")
        
        try: 
            amount = int(response)
            amount = max(0, min(amount, min(market_demand, self.beer_storage)))
            prompt = f"""
            You successfully sold {amount} beer bottles.
            """
        except:

            # Failed to read amount
            print("\n⚠️  WARNING: Problem with reading market sale  (decide market sale) ⚠️")
            print("It should be a number")
            print(response)
            response1 = input("Continue? (y/n): ")
            if response1.lower() != 'y':                    
                self.game_active = False
                return 0

            amount = min(market_demand, self.beer_storage)  # Default: sell maximum possible
        
        return amount

# Game Orchestrator
class BeerGame:
    def __init__(self, params: GlobalParameters, llm_interfaces: Dict[str, Any]):
        self.params = params
        self.database = Database()
        self.current_round = 0
        
        # Initialize companies
        self.factory = Factory("Factory", params, llm_interfaces.get("factory"))
        self.wholesale = Wholesale("Wholesale", params, llm_interfaces.get("wholesale"))
        self.retail = Retail("Retail", params, llm_interfaces.get("retail"))

        models_used = {}
        for role, llm in llm_interfaces.items():
            if llm:
                models_used[role] = {
                    "api_type": llm.api_type,
                    "model": llm.model,
                    "temperature": llm.temperature,
                    "max_tokens": llm.max_tokens,
                    "system_prompt": llm.system_prompt,
                }
        self.database.log_models_used(models_used)

        self.companies = [self.factory, self.wholesale, self.retail]
        self.company_map = {c.name: c for c in self.companies}
        
        # Track deliveries for next round
        self.pending_deliveries = {
            "Factory->Wholesale": 0,
            "Wholesale->Retail": 0
        }
        
        self.game_active = True
        self.contract_id_counter = 0

    def initialize_game(self):
        """Send game rules to all agents at the start"""
        
        # Initialize each company with the rules
        for company in self.companies:
            company.initialize_agent()
    
    def handle_contracts_and_orders(self, supplier: Company, buyer: Company, round_num: int) -> Tuple[List[Tuple[int, float, str, Any]], float]:
        """Handle all contracts and one-time orders between two companies
        Returns: (orders_list, total_cost) where orders_list contains tuples of (amount, price_per_unit, order_type, source)
        """
        orders = []  # List of (amount, price_per_unit, order_type, source_info)
        total_cost = 0
        
        # First, check existing backlog (will be handled with one-time orders priority)
        if buyer.name in supplier.backlog:
            backlog_amount = supplier.backlog[buyer.name]
            orders.append((backlog_amount, 0.0, "backlog", None))
            del supplier.backlog[buyer.name]        
        
        # Process active contracts - separate by type for priority
        flexible_orders = []
        fixed_orders = []
        
        for contract in buyer.contracts:
            if contract.parties[0] == supplier.name and contract.is_active(round_num):
                if contract.contract_type == ContractType.FLEXIBLE:
                    # Buyer chooses amount for flexible contract
                    min_amt, max_amt = contract.get_flexible_range(self.params.flexibility_percentage)
                    prompt = f"Flexible contract: Choose amount between {min_amt} and {max_amt}"
                    response = buyer.get_llm_response(prompt, f"Answer ONLY with a number between {min_amt} and {max_amt}")
                    try: 
                        amount = int(response)
                        amount = max(min_amt, min(max_amt, amount))
                    except:

                        # Failed to read amount
                        print("\n⚠️  WARNING: Problem with reading flexible contract amount (handle_contracts_and_orders) ⚠️")
                        print("It has to be a number: ")
                        print(response)
                        
                        response1 = input("Continue? (y/n): ")
                        if response1.lower() != 'y':                    
                            self.game_active = False
                            return 0

                        amount = contract.amount
                    
                    flexible_orders.append((amount, contract.price_per_unit, "flexible", contract))
                    total_cost += amount * contract.price_per_unit
                else:
                    fixed_orders.append((contract.amount, contract.price_per_unit, "fixed", contract))
                    total_cost += contract.amount * contract.price_per_unit
        
        # One-time order negotiation
        one_time_amount = 0
        one_time_price = 0
        
        # Calculate what's already coming
        total_coming = sum(amt for amt, _, _, _ in flexible_orders + fixed_orders)
        
        # Include backlog captured earlier (we popped it into 'orders' and deleted the dict entry)
        if 'backlog_amount' in locals():
            total_coming += backlog_amount


        # FIXME I don't understand next piece
        # What for it is here? 

        # Calculate what's coming next round from contracts/backlog
        total_coming_next = 0
        for c in buyer.contracts:
            if c.parties[0] == supplier.name and c.is_active(round_num + 1):
                # For flexible we don't know next-round choice yet; use base amount as expectation
                total_coming_next += c.amount

        info = f"You are going to receive {total_coming} units next round from contracts/backlog."
        buyer.add_message_to_buffer(info)

        info = f"""
        You already owe {total_coming} units this round (contracts + backlog)."""
        supplier.add_message_to_buffer(info)

        prompt = "Do you want to order additional beer at default price? Answer: Yes/No"
        response = buyer.get_llm_response(prompt, "Answer ONLY: Yes or No")
        
        if "yes" in response.lower():
            prompt = "How many additional units do you want to order?"
            response = buyer.get_llm_response(prompt, "Answer ONLY with a number")
            try:
                additional = int(response)
                additional = max(0, additional)
                
                # Negotiate with supplier

                total_if_accept = total_coming + additional
                maximum_delivery = supplier.beer_storage - total_coming
                supplier_prompt = f"""{buyer.name} requests a one-time additional {additional} units at the default price.
                
                IF YOU ACCEPT, your total to deliver this round becomes:
                
                {total_coming} + {additional} = {total_if_accept}
                
                Your current inventory: {supplier.beer_storage}.
                Reminder: one-time orders are in addition to existing obligations (they do not replace them).
                """
                
                if maximum_delivery > 0:
                    supplier_prompt +=f"The maximum amount you can additionaly send THIS turn is {maximum_delivery}."
                else:
                    supplier_prompt +=" Therefore you can't deliver any additional beer this round. If you accept the proposal the whole amount will go the your backlog and you will have to pay fee for each bottle."

                
                supplier_prompt +=f"""
                Answer ONLY one of:
                - Yes        (accept full {additional})
                - No         (reject one-time order)
                - <number>   (counter-offer that many one-time units)
                """

                supplier_response = supplier.get_llm_response(supplier_prompt, "")
                
                if "yes" in supplier_response.lower():
                    one_time_amount = additional
                    prompt = f"{supplier.name} agreed"
                    buyer.add_message_to_buffer(prompt)
                elif "no" in supplier_response.lower():
                    prompt = f"{supplier.name} rejected your proposal"
                    buyer.add_message_to_buffer(prompt)
                else:
                    try:
                        counter_amount = int(supplier_response)
                        
                        # Ask buyer if they accept the counter-offer
                        buyer_prompt = f"{supplier.name} counter-offers {counter_amount} units at default price. Accept?"
                        buyer_response = buyer.get_llm_response(buyer_prompt, "Answer ONLY: Yes or No")
                        
                        if "yes" in buyer_response.lower():
                            one_time_amount = counter_amount
                            prompt = f"{buyer.name} agreed"
                            supplier.add_message_to_buffer(prompt)
                        else: 
                            prompt = f"{buyer.name} rejected your proposal"
                            supplier.add_message_to_buffer(prompt)
                    except:

                        print("\n⚠️  WARNING: Problem with reading on-spot COUNTER-order amount (handle_contracts_and_orders) ⚠️")
                        print("It has to be a number")
                        print(response)

                        pass  # Failed to parse counter amount, no deal
                        
                if one_time_amount > 0:
                    price_index = 0 if supplier.company_type == CompanyType.FACTORY else 1
                    one_time_price = self.params.base_prices[price_index]
                    orders.append((one_time_amount, one_time_price, "one_time", None))
                    total_cost += one_time_amount * one_time_price
            except:
                # Failed to read amount
                print("\n⚠️  WARNING: Problem with reading on-spot order amount (handle_contracts_and_orders) ⚠️")
                print("It has to be a number")
                print(response)
                
                response1 = input("Continue? (y/n): ")
                if response1.lower() != 'y':                    
                    self.game_active = False
                    return 0
                pass
        
        # Sort orders by priority (revenue maximization for supplier)
        # 1. Flexible contracts (highest price, 1.2x)
        # 2. Fixed contracts (medium price, 1.1x)
        # 3. One-time and backlog mixed by age (base price, 1.0x)
        
        priority_orders = []
        
        # Add flexible contracts first (sorted by contract start date for ties)
        flexible_orders.sort(key=lambda x: x[3].start_round if x[3] else 0)
        priority_orders.extend(flexible_orders)
        
        # Add fixed contracts second (sorted by contract start date)
        fixed_orders.sort(key=lambda x: x[3].start_round if x[3] else 0)
        priority_orders.extend(fixed_orders)
        
        # Add one-time and backlog (backlog is older so goes first)
        for order in orders:
            if order[2] in ["backlog", "one_time"]:
                priority_orders.append(order)
        
        return priority_orders, total_cost
    
    def execute_delivery(self, supplier: Company, buyer: Company, 
                    orders: List[Tuple[int, float, str, Any]], round_num: int) -> int:
        """Execute delivery from supplier to buyer with priority-based fulfillment
        Args:
            orders: Priority-sorted list of (amount, price_per_unit, order_type, source_info)
        Returns:
            Total amount delivered (for pending_deliveries tracking)
        """
        remaining_beer = supplier.beer_storage
        total_delivered = 0
        total_payment = 0
        
        # Track if there's a shortage
        total_needed = sum(amount for amount, _, _, _ in orders)
        has_shortage = total_needed > supplier.beer_storage

        # Determine backlog fee based on supplier type
        if supplier.company_type == CompanyType.FACTORY:
            backlog_fee_per_unit = self.params.base_prices[0]
        elif supplier.company_type == CompanyType.WHOLESALE:
            backlog_fee_per_unit = self.params.base_prices[1]
        
        # Check for irrational behavior: one-time orders during shortage
        if has_shortage:
            has_one_time = any(order_type == "one_time" for _, _, order_type, _ in orders)
            if has_one_time:
                print("\n⚠️  WARNING: IRRATIONAL BEHAVIOR DETECTED ⚠️")
                print(f"Supplier: {supplier.name} | Buyer: {buyer.name}")
                print(f"Storage: {supplier.beer_storage} | Total needed: {total_needed}")
                print("One-time orders exist during shortage situation!")
                print("This may indicate prompt engineering issues.")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    self.game_active = False
                    return 0
        
        # Process orders by priority
        for amount, price_per_unit, order_type, source_info in orders:
            
            if order_type in ["fixed", "flexible"]:
                contract = source_info
                
                if remaining_beer < amount:
                    shortage = amount - max(0, remaining_beer)  # How much we can't deliver
                    available = max(0, remaining_beer)  # How much we could deliver
                    
                    prompt = f"""
                    Contract requires {amount} units, you have {available} available.
                    OPTIONS (answer ONLY 1, 2, or 3):
                    1. Deliver {available} units, add {shortage} to backlog
                        - Backlog fee per round: ${backlog_fee_per_unit} per unit
                    2. Break contract now and deliver nothing
                        - Pay fine: ${contract.fine}
                    3. Propose mutual cancellation (no penalties if buyer agrees) 
                        - No penalties if buyer ACCEPTS; if buyer REFUSES, you must then choose 1 or 2                  
                    """
                    response = supplier.get_llm_response(prompt, "Answer only: 1, 2 or 3")
                    
                    if "3" in response:
                        # Propose mutual cancellation
                        buyer_prompt = f"""
                        {supplier.name} cannot fulfill contract ({shortage} units short) and proposes mutual cancellation.
                        If you agree, contract ends with no penalties.
                        If you refuse, they must choose to deliver partial or break with fine.
                        Accept mutual cancellation?
                        """
                        buyer_response = buyer.get_llm_response(buyer_prompt, "Answer only: Yes or No")
                        
                        if "yes" in buyer_response.lower():
                            # Mutual cancellation - no penalties
                            supplier.contracts.remove(contract)
                            buyer.contracts.remove(contract)
                            self.database.log_contract(contract, "MUTUALLY_CANCELLED")
                            continue  # Skip this order entirely
                        else:
                            # Buyer refused, ask supplier again without option 3
                            prompt = f"""
                            Buyer refused mutual cancellation. You must choose:
                            1. Deliver {available} units, add {shortage} to backlog (fee: ${shortage * backlog_fee_per_unit}/round)
                            2. Break contract and pay fine: ${contract.fine} (deliver nothing)
                            """
                            response = supplier.get_llm_response(prompt, "Answer only: 1 or 2")
                    

                    if "2" in response:
                        # Break contract - deliver nothing, pay fine
                        supplier.money -= contract.fine
                        buyer.money += contract.fine
                        supplier.contracts.remove(contract)
                        buyer.contracts.remove(contract)
                        self.database.log_contract(contract, "BROKEN")
                        continue  # Skip this order entirely
                
            delivered = min(amount, remaining_beer)
            payment = delivered * price_per_unit
            
            # Execute payment
            buyer.money -= payment
            supplier.money += payment
            total_payment += payment
            
            # Update remaining beer
            remaining_beer -= delivered
            total_delivered += delivered
            
            # Log transaction
            self.database.log_transaction(
                round_num, supplier.name, buyer.name, 
                delivered, payment, f"delivery_{order_type}"
            )
            
            # Handle shortage for this order
            if delivered < amount:
                shortage = amount - delivered
                
                if buyer.name not in supplier.backlog:
                    supplier.backlog[buyer.name] = 0
                supplier.backlog[buyer.name] += shortage
                
        
        # Clear any fulfilled backlog
        if buyer.name in supplier.backlog and order_type == "backlog":
            supplier.backlog[buyer.name] = max(0, supplier.backlog[buyer.name] - total_delivered)
            if supplier.backlog[buyer.name] == 0:
                del supplier.backlog[buyer.name]
            
        # Pay backlog fees for remaining backlog
        if buyer.name in supplier.backlog:
            backlog_fee = supplier.backlog[buyer.name] * backlog_fee_per_unit
            supplier.money -= backlog_fee
            buyer.money += backlog_fee
            self.database.log_transaction(
                round_num, supplier.name, buyer.name,
                0, backlog_fee, "backlog_fee"
            )
        
        # Update supplier's beer storage
        supplier.beer_storage -= total_delivered
        
        return total_delivered
    
    def run_round(self):
        """Execute one complete round of the game"""
        self.current_round += 1
        round_num = self.current_round
        
        print(f"\n=== Round {round_num} ===")

        # Track starting values
        for company in self.companies:
            company.track_round_start()
        
        # 1. Retail and wholesali receive beer from previous round
        self.retail.beer_storage += self.pending_deliveries["Wholesale->Retail"]
        self.pending_deliveries["Wholesale->Retail"] = 0

        self.wholesale.beer_storage += self.pending_deliveries["Factory->Wholesale"]
        self.pending_deliveries["Factory->Wholesale"] = 0
        
        # 2. Calculate and log market demand
        market_demand = market_demand_function(round_num)
        self.database.log_market_demand(round_num, market_demand)
        
        # 3. Inform all companies of their status
        for company in self.companies:
            status = f"""
            Round {round_num} Status:
            Money: ${company.money:.2f}
            Beer in storage: {company.beer_storage}
            Active contracts: {len([c for c in company.contracts if c.is_active(round_num)])}
            Backlog owed: {sum(company.backlog.values())}
            
            """
            company.add_message_to_buffer(status)
            
            if company.name != "retail":
                supplier_contracts = [c for c in company.contracts if c.is_active(round_num) and c.parties[0] == company.name]

                # Fixed obligations are known now; flexible will be chosen later this round, so show a range
                due_now_fixed = sum(c.amount for c in supplier_contracts if c.contract_type == ContractType.FIXED)
                flex_min = sum(int(c.amount * (1 - self.params.flexibility_percentage)) for c in supplier_contracts if c.contract_type == ContractType.FLEXIBLE)
                flex_max = sum(int(c.amount * (1 + self.params.flexibility_percentage)) for c in supplier_contracts if c.contract_type == ContractType.FLEXIBLE)

                if flex_min or flex_max:
                    obligations_line = f"Obligations this round for you to deliver: fixed contracts ={due_now_fixed}, flexible contracts: from {flex_min} to {flex_max} depending on buyer decition"
                else:
                    obligations_line = f"Obligations this round for you to deliver: fixed={due_now_fixed}"

                company.add_message_to_buffer(f"{obligations_line}. Backlog owed: {sum(company.backlog.values())}.")


            due_now = sum(c.amount for c in company.contracts if c.is_active(round_num))
            company.add_message_to_buffer(f"Obligations this round (contracts only): {due_now}. Backlog owed: {sum(company.backlog.values())}.")
        
        # 4. Retail-Wholesale interaction
        self.retail.add_message_to_buffer(f"Market demand this round: {market_demand}")
        
        # Call option
        prompt = "Do you want to talk with Wholesale?"
        response = self.retail.get_llm_response(prompt, "Answer only: Yes or No")
        
        if "yes" in response.lower():
            conversation = self.retail.call_neighbor(self.wholesale, self.params.max_messages_per_call)
            self.database.log_dialogue(round_num, "Retail", "Wholesale", conversation)
        
        # Contract negotiation
        if self.params.contracts_enabled:
            prompt = "Do you want to propose a contract to Wholesale? Answer: Yes/No"
            response = self.retail.get_llm_response(prompt, "Answer only: Yes or No")
            
            if "yes" in response.lower():
                prompt = "Specify contract, USE THE FOLLOWING FORMAT: type(fixed/flexible),amount,length,fine"
                response = self.retail.get_llm_response(prompt, "type,amount,length,fine (Example: fixed,100,10,500). DON'T WRITE ANYTHING ELSE.")
                
                # Parse the proposal
                try:
                    parts = response.split(',')
                    if len(parts) == 4:
                        contract_type = parts[0].strip().lower()
                        amount = int(parts[1].strip())
                        length = int(parts[2].strip())
                        fine = float(parts[3].strip())
                        wrong_format = False
                        
                        # Calculate price based on type and direction
                        base_price = self.params.base_prices[1]  # Wholesale->Retail price
                        if contract_type == "flexible":
                            price = base_price * self.params.contract_multipliers["flexible"]
                        elif contract_type == "fixed":
                            price = base_price * self.params.contract_multipliers["fixed"]
                        else:
                            print(f"Invalid contract type: {contract_type}")
                            error_prompt = "Invalid format. You wrote: " + response + ". However, the right format is type,amount,length,fine (Example: fixed,100,10,500). No contract was proposed.\n"
                            self.retail.add_message_to_buffer(error_prompt)
                            wrong_format = True
                        
                        if wrong_format == False:
                            contract_details = {
                                'type': contract_type,
                                'amount': amount,
                                'length': length,
                                'fine': fine,
                                'price_per_unit': price,
                                'start_round': round_num}
                                
                            # Use the propose_contract method
                            contract = self.retail.propose_contract(self.wholesale, contract_details)
                            if contract:
                                print(f"Contract created: {contract_type}, {amount} units/round for {length} rounds")
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse contract proposal: {response}")
                    response1 = input("Continue? (y/n): ")
                    if response1.lower() != 'y':                    
                        self.game_active = False
                        return 0
                    else:
                        error_prompt = "Invalid format. You wrote: " + response + ". However, the right format is type,amount,length,fine (Example: fixed,100,10,500). No contract was proposed.\n"
                        self.retail.add_message_to_buffer(error_prompt)                        
        
        # 6-7. Handle orders and contracts Retail-Wholesale
        beer_ordered, money_owed = self.handle_contracts_and_orders(self.wholesale, self.retail, round_num)
        
        # 8. Retail sells to market
        #sale_amount = self.retail.decide_market_sale(market_demand) this function gives a possibility to retail to decide how much to sell. But as there are no price fluctuations and penalties for not meeting market demand, the only possible strategy is to sell as much as posible. So the funtion is not used. 
        sale_amount = min(market_demand, self.retail.beer_storage)
        self.retail.beer_storage -= sale_amount
        self.sale_amount = sale_amount
        sale_revenue = sale_amount * self.params.base_prices[2]
        self.retail.money += sale_revenue
        self.retail.add_message_to_buffer(f"{sale_amount} units were authomaticaly sold to the market this round. Sale revenue is {sale_revenue}.\n")
        self.database.log_transaction(round_num, "Market", "Retail", sale_amount, sale_revenue, "market_sale")
        
        # 9-12. Wholesale-Factory interaction (similar structure)
        prompt = "Do you want to talk with Factory?"
        response = self.wholesale.get_llm_response(prompt, "Answer only: Yes or No")
        
        if "yes" in response.lower():
            conversation = self.wholesale.call_neighbor(self.factory, self.params.max_messages_per_call)
            self.database.log_dialogue(round_num, "Wholesale", "Factory", conversation)
        
        if self.params.contracts_enabled:
            prompt = "Do you want to propose a contract to Factory? Answer: Yes/No"
            response = self.wholesale.get_llm_response(prompt, "Answer only: Yes or No")
            
            if "yes" in response.lower():
                prompt = "Specify contract: type(fixed/flexible),amount,length,fine"
                response = self.wholesale.get_llm_response(prompt, "Format: type,amount,length,fine (Example: flexible,150,8,750)")
                
                # Parse the proposal
                try:
                    parts = response.split(',')
                    if len(parts) == 4:
                        contract_type = parts[0].strip().lower()
                        amount = int(parts[1].strip())
                        length = int(parts[2].strip())
                        fine = float(parts[3].strip())
                        
                        # Calculate price based on type and direction
                        base_price = self.params.base_prices[0]  # Factory->Wholesale price
                        if contract_type == "flexible":
                            price = base_price * self.params.contract_multipliers["flexible"]
                        elif contract_type == "fixed":
                            price = base_price * self.params.contract_multipliers["fixed"]
                        else:
                            print(f"Invalid contract type: {contract_type}")
                            price = None
                        
                        if price:
                            contract_details = {
                                'type': contract_type,
                                'amount': amount,
                                'length': length,
                                'fine': fine,
                                'price_per_unit': price,
                                'start_round': round_num  # Starts next round
                            }
                            
                            # Use the propose_contract method
                            contract = self.wholesale.propose_contract(self.factory, contract_details)
                            if contract:
                                print(f"Contract created: {contract_type}, {amount} units/round for {length} rounds")
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse contract proposal: {response}")
                    response1 = input("Continue? (y/n): ")
                    if response1.lower() != 'y':                    
                        self.game_active = False
                        return 0
        
        wholesale_beer_ordered, wholesale_money_owed = self.handle_contracts_and_orders(
            self.factory, self.wholesale, round_num
        )
        
        # 13. Execute Wholesale->Retail delivery
        will_be_delivered = self.execute_delivery(self.wholesale, self.retail, beer_ordered, round_num)
        self.pending_deliveries["Wholesale->Retail"] = will_be_delivered
        

        will_be_delivered = self.execute_delivery(self.factory, self.wholesale, 
                                        wholesale_beer_ordered, round_num)
        self.pending_deliveries["Factory->Wholesale"] = will_be_delivered
        
        # Factory produces for next round
        self.factory.decide_production(round_num)
        self.factory.produce_beer()
        
        # 15. End round checks
        for company in self.companies:
            # Remove excess storage
            company.remove_excess_storage()
            
            # Storage costs
            company.pay_storage_costs()
            
            # Check bankruptcy
            if company.money < 0:
                print(f"Game Over: {company.name} is bankrupt!")
                self.game_active = False
                break

            metrics = company.calculate_round_metrics(round_num)
            company.provide_round_feedback(metrics, round_num)
        
        # Log round state
        states = {c.name: c.get_state(round_num) for c in self.companies}
        self.database.log_round_state(round_num, states)
    
    def run_game(self):
        """Run the complete game"""
        print("Starting Beer Supply Chain Game")
        print(f"Running for {self.params.num_rounds} rounds")

        # Initialize all agents with game rules
        self.initialize_game()
        
        while self.game_active and self.current_round < self.params.num_rounds:
            self.run_round()
        
        print("\nGame Complete!")
        self.print_final_scores()

    
    def print_final_scores(self):
        """Print final game statistics"""
        print("\n=== Final Scores ===")
        i = 0
        for company in self.companies:
            total_value = company.money + (company.beer_storage * self.params.base_prices[i])
            i+=1
            print(f"{company.name}:")
            print(f"  Money: ${company.money:.2f}")
            print(f"  Beer in storage: {company.beer_storage}")
            print(f"  Total value: ${total_value:.2f}")
            if company.name == "wholesale":
                print(f"  Contracts completed: {len([c for c in company.contracts if not c.is_active(self.current_round)])}")
                print(f"  Ongoing contracts: {len([c for c in company.contracts if c.is_active(self.current_round)])}")

    def create_contract(self, supplier: Company, buyer: Company, contract_details: Dict) -> Contract:
        """Create a new contract with a unique ID"""
        contract = Contract(
            contract_id=self.contract_id_counter,
            parties=(supplier.name, buyer.name),
            start_round=contract_details['start_round'],
            length=contract_details['length'],
            amount=contract_details['amount'],
            contract_type=ContractType[contract_details['type'].upper()],
            fine=contract_details['fine'],
            price_per_unit=contract_details['price_per_unit']
        )
        self.contract_id_counter += 1
        
        supplier.contracts.append(contract)
        buyer.contracts.append(contract)
        self.database.log_contract(contract, "CREATED")
        
        return contract

# LLM Interface placeholder
class MockLLM:
    """Mock LLM for testing - replace with actual LLM integration"""
    def generate(self, prompt: str) -> str:
        # Simple mock responses for testing
        if "Yes/No" in prompt or "Yes or No" in prompt:
            return random.choice(["Yes", "No"])
        elif "number" in prompt.lower():
            return str(random.randint(100, 300))
        elif "Accept/Reject/Counter" in prompt:
            return random.choice(["Accept", "Reject", "Counter"])
        else:
            return "Acknowledged"
        
class LLMInterface:
    """
    Unified interface for OpenAI and Anthropic LLMs
    """
    def __init__(self, api_type: str, api_key: str, model: str = None, 
                 temperature: float = 0.7, max_tokens: int = 150,
                 system_prompt: str = None, company_name: str = None):
        """
        Initialize LLM interface
        
        Args:
            api_type: 'openai' or 'anthropic'
            api_key: API key for the chosen service
            model: Model name (optional, uses defaults if not specified)
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for the company role
        """
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or ""
        self.name = company_name
        
        if api_type == "openai":
            if openai is None:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
        elif api_type == "anthropic":
            if anthropic is None:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model or "claude-3-5-haiku-20241022"
        elif api_type == "together":
            # Together is OpenAI-compatible: just set the base_url
            self.client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
            # Pick any Together chat model here; you can change this per company later
            self.model = model or "meta-llama/Llama-3-8b-chat-hf"
        else:
            raise ValueError(f"Unsupported API type: {api_type}. Choose 'openai' or 'anthropic'")
    
    def _update_token_log(self, prompt_tokens: int, completion_tokens: int):
        key = f"{self.api_type}:{self.model}"
        try:
            data = {}
            if os.path.exists(TOKEN_LOG_FILE):
                with open(TOKEN_LOG_FILE, "r") as f:
                    try:
                        import json as _json
                        data = _json.load(f)
                    except Exception:
                        data = {}
            entry = data.get(key, {"prompt_tokens": 0, "completion_tokens": 0})
            entry["prompt_tokens"] += int(prompt_tokens or 0)
            entry["completion_tokens"] += int(completion_tokens or 0)
            data[key] = entry
            with open(TOKEN_LOG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Don't break the game if logging fails
            pass

    def clean_response(self, response: str) -> str:
        """Remove common formatting artifacts from LLM responses"""
        # Remove asterisks (markdown bold/italic)
        cleaned = response.replace('*', '')
        
        # Also remove other common formatting issues
        cleaned = cleaned.replace('`', '')  # Remove backticks (code formatting)
        cleaned = cleaned.replace('#', '')  # Remove headers
        cleaned = cleaned.replace('_', '')  # Remove underscores (italic)
        cleaned = cleaned.replace('~', '')  # Remove strikethrough
        
        # Strip extra whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            String response from the LLM
        """
        print("------------------------------\n")
        print("------------------------------\n")
        print("Prompt:\n ")
        print(prompt)
        print("------------------------------\n")

        try:
            if self.api_type in ("openai", "together"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                response = self.clean_response(response)

                print(self.name + ":\n ")
                print(response.choices[0].message.content.strip())
                print("------------------------------\n")

                # OpenAI/Together usage -> prompt_tokens / completion_tokens
                usage = getattr(response, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                ct = getattr(usage, "completion_tokens", 0) if usage else 0
                self._update_token_log(pt, ct)

                return response.choices[0].message.content.strip()
                
            elif self.api_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                response = self.clean_response(response)

                usage = getattr(response, "usage", None)
                pt = getattr(usage, "input_tokens", 0) if usage else 0
                ct = getattr(usage, "output_tokens", 0) if usage else 0
                self._update_token_log(pt, ct)

                print(self.name + ":\n ")
                print(response.content[0].text.strip())
                print("------------------------------\n")

                return response.content[0].text.strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback to simple defaults for critical decisions
            if "Yes/No" in prompt or "Yes or No" in prompt:
                return "No"  # Conservative default
            elif "number" in prompt.lower():
                return "100"  # Safe middle-ground amount
            else:
                return "Unable to process request"
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.system_prompt = prompt
    
    def set_temperature(self, temperature: float):
        """Update temperature setting"""
        self.temperature = max(0, min(1, temperature))
    
    def set_max_tokens(self, max_tokens: int):
        """Update max tokens setting"""
        self.max_tokens = max(1, max_tokens)

    
def market_demand_function(round_num: int) -> int:
    """Calculate market demand for current round
    Consider changing market_info prompt in the initialize_agent method of the company class to provide agents information about market behaviour."""
    # Simple sinusoidal demand with some randomness
    base_demand = 300
    """seasonal = int(100 * math.sin(round_num * 2 * math.pi / 52))
    random_factor = random.randint(-50, 50)
        
    # Special events/crises
    if round_num in [10, 25, 40]:  # Crisis rounds
        return base_demand // 2
    elif round_num in [15, 30, 45]:  # Boom rounds
        return base_demand * 2
        
    return max(0, base_demand + seasonal + random_factor)"""
    return base_demand
    

# Factory function to create company-specific LLM interfaces
def create_company_llm(company_name: str, api_type: str, api_key: str, 
                       model: str = None, **kwargs) -> LLMInterface:
    """
    Create an LLM interface with company-specific system prompts
    
    Args:
        company_type: 'factory', 'wholesale', or 'retail'
        api_type: 'openai' or 'anthropic'
        api_key: API key
        model: Optional model name
        **kwargs: Additional LLMInterface parameters
    
    Returns:
        Configured LLMInterface
    """

    system_prompt = f"""YOU ARE the {company_name} company in a beer supply chain game. You are NOT an AI assistant - you ARE the decision-maker for {company_name}."""

    system_prompt

    return LLMInterface(api_type, api_key, model, system_prompt=system_prompt, company_name = company_name, **kwargs)

# Main execution
if __name__ == "__main__":
    
    #api_key = os.getenv("OPENAI_API_KEY") 
    #api_key = os.getenv("ANTHROPIC_API_KEY")
    api_key = os.getenv("TOGETHER_API_KEY")
    """
    togetherAI models I use: 
    - meta-llama/Llama-3.3-70B-Instruct-Turbo - too week to grab the game, very poor perfomance, hallucinations, but the cheapest. Many bugs were found using it.
    - nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    - deepseek-ai/DeepSeek-R1-Distill-Llama-70B - twice more expencive in terms of tokens, but it's thinking! That's a lot of tokens. FIXME reasoning is not implemented.
    """
    together_model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    
    # Create LLM interfaces for each company
    llm_interfaces = {
        "factory": create_company_llm("factory", "together", api_key, model=together_model, temperature=0.7),
        "wholesale": create_company_llm("wholesale", "together", api_key, model=together_model, temperature=0.7),
        "retail": create_company_llm("retail", "together", api_key, model=together_model, temperature=0.7)
    }
    
    # Alternative: Using Anthropic
    # anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    # llm_interfaces = {
    #     "factory": create_company_llm("factory", "anthropic", anthropic_key),
    #     "wholesale": create_company_llm("wholesale", "anthropic", anthropic_key),
    #     "retail": create_company_llm("retail", "anthropic", anthropic_key)
    # }

    # You can also mix APIs if desired
    # llm_interfaces = {
    #     "factory": create_company_llm("factory", "openai", openai_key),
    #     "wholesale": create_company_llm("wholesale", "anthropic", anthropic_key),
    #     "retail": create_company_llm("retail", "openai", openai_key)
    # }
    
    print("LLM interfaces created successfully!")
    
    # Create and run game
    params = GlobalParameters()
    game = BeerGame(params, llm_interfaces)
    game.run_game()

    # Access the database after game completes
    database = game.database
    
    # Export with custo filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"game_results_{timestamp}.json"
    database.export_to_json(filename)
    print(f"Results exported to {filename}")
    
    # Optional: Additional analysis
    print(f"\nGame Statistics:")
    print(f"Total transactions: {len(database.transactions)}")

    created_ids = {e['contract']['contract_id'] for e in database.contracts_log if e['action'] == 'CREATED'}
    canceled_ids = {e['contract']['contract_id'] for e in database.contracts_log if e['action'] == 'MUTUALLY_CANCELLED'}
    broken_ids = {e['contract']['contract_id'] for e in database.contracts_log if e['action'] == 'BROKEN'}
    active_ids = created_ids - canceled_ids - broken_ids

    print(f"Total contracts created: {len(created_ids)}")
    print(f"Total contracts canceled: {len(canceled_ids)}")
    print(f"Total contracts broken: {len(broken_ids)}")
    print(f"Active contracts at end: {len(active_ids)}")

    print(f"Total dialogues: {len(database.dialogues)}")
    print(f"Rounds played: {game.current_round}")