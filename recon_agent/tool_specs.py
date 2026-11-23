"""JSON schemas for the tools, in Anthropic tool-use format."""

TOOL_SPECS = [
    {"name": "get_bank_line", "description": "Get a bank statement line by id.",
     "input_schema": {"type": "object", "properties": {"line_id": {"type": "string"}}, "required": ["line_id"]}},
    {"name": "find_customer", "description": "Fuzzy-match bank description text to customers. Returns candidates with a 0-1 score.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "search_open_invoices", "description": "Search unallocated open invoices. All filters optional.",
     "input_schema": {"type": "object", "properties": {
         "customer_id": {"type": "string"},
         "invoice_refs": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['SI-1003']"},
         "amount": {"type": "number"}}}},
    {"name": "find_invoice_combinations", "description": "Find sets of 2-3 open invoices for a customer that sum exactly to an amount.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}, "amount": {"type": "number"}},
                      "required": ["customer_id", "amount"]}},
    {"name": "propose_match", "description": "Propose matching the bank line to invoices. Guardrails may reject it with a reason; if so, either fix the proposal or flag for review.",
     "input_schema": {"type": "object", "properties": {
         "line_id": {"type": "string"},
         "invoice_nos": {"type": "array", "items": {"type": "string"}},
         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
         "reasoning": {"type": "string"}},
         "required": ["line_id", "invoice_nos", "confidence", "reasoning"]}},
    {"name": "flag_for_review", "description": "Send the bank line to a human accountant with a short reason.",
     "input_schema": {"type": "object", "properties": {"line_id": {"type": "string"}, "reason": {"type": "string"}},
                      "required": ["line_id", "reason"]}},
]
