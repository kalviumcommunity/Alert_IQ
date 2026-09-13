# End-to-End Demo Evidence

This sample records the expected shape of the final upload-to-answer smoke run.

```text
[1/3] Backend healthy
[2/3] Indexed: refund-policy.md
[3/3] Grounded answer
Customers can request a refund within 14 days of purchase.

Sources:
[1] refund-policy.md · chunk refund-policy-03 · score 0.910
```

The demo document is `data/samples/refund-policy.md`. The exact generated answer and retrieval score may vary by model/provider and current index state; the acceptance criteria are successful upload/indexing, a grounded answer, and inspectable citation metadata.
