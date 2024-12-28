import os
from flask import request
import requests
from totoapicontroller.model.ExecutionContext import ExecutionContext

class ExpensesAPI: 
    
    endpoint: str
    exec_context: ExecutionContext
    auth_header: str
    
    def __init__(self, exec_context: ExecutionContext, auth_header: str) -> None:
        # Set the endpoint data
        self.endpoint = os.environ.get('EXPENSES_API_ENDPOINT')
        
        # Set passed vars
        self.exec_context = exec_context
        self.auth_header = auth_header
        
    def get_last_supermarket_expenses(self, max_results: int = 1) : 
        """
        Retrieves the last supermarket expenses
        """
        response = requests.get(
            f"{self.endpoint}/expenses?sortDate=true&sortDesc=true&category=SUPERMERCATO&maxResults={max_results}", 
            headers={
                "Accept": "application/json", 
                "Authorization": self.auth_header, 
                "x-correlation-id": self.exec_context.cid
            }
        )
        
        if response.status_code == 200: 
            return response.json()
        else: 
            self.exec_context.logger.log(self.exec_context.cid, f"ExpensesAPI generated an ERROR with code [{response.status_code}] and message [{response.text}]")
            return None
        
