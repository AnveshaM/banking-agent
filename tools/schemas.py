"""
tools/schemas.py — Layer 1: Pydantic v2 input schemas.

Every tool input is a Pydantic model. Two effects:
  1. Validation runs before the function body — bad args rejected immediately.
  2. ADK reads the schema to build the tool spec the model sees, so the model
     knows the correct format before it ever makes a call.

This file has no project imports — intentional. Schemas are a dependency
of everything, a dependent of nothing.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LookupInvoiceInput(BaseModel):
    invoice_id: str = Field(
        ...,
        description="Invoice ID in format INV-NNNNN (five digits). Example: INV-00123.",
    )

    @field_validator("invoice_id")
    @classmethod
    def valid_format(cls, v: str) -> str:
        if not re.fullmatch(r"INV-\d{5}", v):
            raise ValueError(
                f"'{v}' is not valid. Format: INV-NNNNN (five digits, e.g. INV-00123)."
            )
        return v


class ProposeRefundInput(BaseModel):
    invoice_id: str = Field(
        ...,
        description="Invoice to refund. Must exist and have status 'paid'.",
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Human-readable reason for the refund. Min 10 characters.",
    )
    idempotency_key: str = Field(
        ...,
        description=(
            "Unique key for this proposal. Format: propose-{invoice_id}-{session_id}. "
            "Replaying the same key returns the original proposal without re-queuing."
        ),
    )
    customer_message: str = Field(
        ...,
        min_length=10,
        description="What you would tell the customer about this refund.",
    )

    @field_validator("invoice_id")
    @classmethod
    def valid_format(cls, v: str) -> str:
        if not re.fullmatch(r"INV-\d{5}", v):
            raise ValueError(f"'{v}' is not valid. Format: INV-NNNNN.")
        return v


class SearchKBInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="The user's question in their own words.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of results. Default 3, max 5.",
    )


class RememberCustomerInput(BaseModel):
    customer_id: str = Field(
        ...,
        description="Customer ID to pin for this session. Format: CUST-NNNN.",
    )


class CheckServiceStatusInput(BaseModel):
    service: Literal["api", "dashboard", "webhooks", "payments"] = Field(
        ...,
        description="Which service to check.",
    )
