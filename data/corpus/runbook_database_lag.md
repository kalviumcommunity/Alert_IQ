# Database Replica Latency Triage Runbook (DB-RB-402)

## Overview
This runbook provides emergency mitigation procedures when PostgreSQL read replica replication delay exceeds 500ms or connection pools reach 90% utilization.

## Immediate Triage Steps
1. **Inspect Active Connections**: Run `SELECT pid, state, query_start, query FROM pg_stat_activity WHERE state != 'idle'` to find blocking queries.
2. **Terminate Stalled Queries**: Execute `SELECT pg_terminate_backend(pid)` for transactions running longer than 30 minutes.
3. **Traffic Shedding**: Divert non-critical analytical queries to secondary replica pools.
4. **Escalate**: If replication lag exceeds 1500ms for over 10 minutes, page the Database Reliability SRE lead.
