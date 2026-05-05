# WebGuard MCP — Full Agent Implementation Brief

## 1. Product Name

**WebGuard MCP**

## 2. Product Definition

WebGuard MCP is a production-grade, intelligent MCP (Model Context Protocol) server that monitors website performance, detects bugs and anomalies in real time, and automatically generates and deploys fixes using the Claude AI SDK.

It is not a simple uptime checker or a static alerting tool. WebGuard MCP is an active, reasoning system that understands your codebase, investigates root causes autonomously, proposes or applies targeted code fixes, and validates that each fix actually resolves the issue before it reaches production.

The system exposes its capabilities through the MCP standard so it integrates natively with Claude.ai and any MCP-compatible client, allowing developers to query site health, trigger analysis, approve fixes, and review history through natural language conversation.

## 3. Main Product Goal

Build a complete, production-ready MCP server that delivers the full monitoring and automated remediation experience.

The main operational loop is:

**Monitor website → Detect anomaly or error → Gather context → Claude analyses root cause → Generate fix → Test in isolation → Deploy safely → Validate resolution → Repeat**

Everything in this implementation must support this loop.

Do not add unnecessary social features, billing systems, complex admin portals, or unrelated tooling. The product should feel focused, reliable, and genuinely autonomous.

## 4. Final Scope

The implementation must include:

- Full MCP server with typed tool and resource definitions
- Continuous website performance monitoring
- Real-time error and anomaly detection
- Claude-powered root cause analysis
- Automated code fix generation
- Isolated fix testing in containerised environments
- Safe deployment with rollback capability
- Time-series metrics storage
- Structured incident logging
- Git integration for automated patch management
- Notification routing for alerts and fix reports
- Dashboard API for external consumption
- Multi-site support
- Configurable automation levels
- Production-grade validation and error handling
- Secure credential management
- Clean loading, empty, and error states across all tool responses

## 5. Technology Direction

Use the following stack:

- Node.js with TypeScript
- @modelcontextprotocol/sdk
- @anthropic-ai/sdk
- Fastify
- PostgreSQL with TimescaleDB
- Drizzle ORM
- Redis
- BullMQ
- Playwright
- Lighthouse
- Zod
- Pino
- Vitest
- Docker and Docker Compose
- Octokit
- GitHub Actions

The implementation agent must verify the latest official documentation for each package before implementation and use the latest stable recommended setup.

Do not rely on outdated examples, deprecated APIs, old configuration styles, or stale tutorials.

The agent should use current best practices from official documentation at implementation time.

## 6. Engineering Principles

### 6.1 Production-Ready From the Start

This must not be treated like a demo or proof of concept.

Implement with:

- Strong typing throughout
- Clean validation on all inputs and outputs
- Secure credential handling
- Server-side secrets only
- Proper error boundaries
- Meaningful structured logging
- Sensible caching strategies
- Clean reusable service modules
- Clear separation of concerns between monitoring, analysis, and deployment layers
- Idempotent fix operations where possible

### 6.2 Do Not Overengineer

Do not add unnecessary ML pipelines, billing systems, social features, advanced visualisation, or mobile client code unless absolutely required for the core product to work.

Keep the product lean but complete.

### 6.3 Build for Future Extension Without Building It Now

The architecture should be clean enough to support future clients and integrations without requiring a full rewrite.

This means:

- Do not tightly couple monitoring logic to the MCP transport layer.
- Keep business logic in dedicated service classes.
- Use predictable, stable response shapes for all tools and resources.
- Normalise external data before it enters the system.
- Avoid exposing raw Playwright, Lighthouse, or external API structures across the codebase.

### 6.4 External API and Credential Safety

API keys, GitHub tokens, database credentials, and the Anthropic API key must never be exposed to clients or logged carelessly.

All secrets must be loaded through environment variables and accessed only from the server layer.

The system must never log raw credential values even during debug states.

### 6.5 Automation Safety First

The system has the ability to deploy code changes to production repositories. This is a high-trust operation and must be handled with extreme care.

Every automated deployment must:

- Pass isolated container tests before deploying
- Require a rollback plan to exist before the fix is applied
- Log the complete diff and rationale
- Support immediate rollback on degradation signals
- Never apply a fix that was not generated and validated by the defined pipeline

Manual approval mode must be supported and must be the default for new site configurations.

## 7. Product Experience Principles

WebGuard MCP should feel:

- Trustworthy
- Precise
- Fast to respond
- Calm under pressure
- Autonomous without being reckless
- Easy to configure and reason about
- Transparent about what it is doing and why

WebGuard MCP should not feel:

- Like a black box that silently changes code
- Like a noisy alerting tool that cries wolf
- Like an overcomplicated enterprise platform
- Like a half-finished prototype
- Like a system that hides its reasoning

## 8. MCP Server Architecture

### 8.1 Server Structure

The MCP server must be implemented as a clean TypeScript project following the official MCP SDK structure.

Use the stdio transport for local/Claude.ai integration and HTTP/SSE transport for remote integration.

Both transports must be supported through configuration.

The server must start cleanly, register all tools and resources, and expose a health endpoint for monitoring its own availability.

### 8.2 Tool Registration

All tools must be defined with:

- Unique, descriptive name
- Clear human-readable description
- Zod-validated input schema
- Typed return shape
- Proper error handling that returns structured MCP errors, not raw exceptions

### 8.3 Resource Registration

Resources must expose live and historical data through:

- URI-based addressing
- Consistent JSON structure
- Human-readable descriptions
- Appropriate content types

### 8.4 Capability Boundaries

The MCP server must clearly separate:

- Data collection (monitoring layer)
- Analysis (Claude SDK layer)
- Action execution (deployment layer)
- Data access (resource layer)

These layers must not be mixed within a single tool handler.

## 9. MCP Tools Specification

### 9.1 monitor_website

**Purpose:** Register a website for continuous monitoring or update an existing site configuration.

**Input:**
- url (required)
- name (optional display label)
- checkIntervalSeconds (optional, default 60)
- metrics (optional array: performance, errors, resources, uptime)
- automationLevel (optional: alert_only, suggest, auto_safe, auto_full)
- notifyChannels (optional array of channel identifiers)

**Behaviour:**

Add the site to the monitored sites registry. Begin scheduling health checks according to the configured interval. Store initial baseline metrics on first successful check.

If the site already exists, update the configuration and reset the baseline if the URL changed.

**Returns:**

Structured site summary including site ID, configuration, and initial status.

### 9.2 get_site_health

**Purpose:** Return the current health status of a monitored site.

**Input:**
- siteId (required)

**Behaviour:**

Retrieve the most recent metrics snapshot, active incidents, and overall health score for the specified site.

**Returns:**

Health object including current status, response time, error rate, active incidents, and trend direction over the last hour.

### 9.3 list_sites

**Purpose:** List all registered monitored sites and their current statuses.

**Input:**

No required input. Optional filter by status.

**Returns:**

Array of site summaries ordered by most recently active incident first.

### 9.4 analyze_performance

**Purpose:** Trigger a deep performance analysis for a specific site over a given time range.

**Input:**
- siteId (required)
- timeRangeHours (optional, default 24)
- focus (optional: response_time, error_rate, memory, database, all)

**Behaviour:**

Pull relevant metrics from the time-series store. Retrieve associated error logs. Pass context to Claude for root cause analysis. Return the structured analysis result.

**Returns:**

Analysis object including identified issues, severity levels, root cause summary, and recommended actions.

### 9.5 detect_bugs

**Purpose:** Scan active incidents and error patterns to identify bugs requiring attention.

**Input:**
- siteId (required)
- severity (optional array: critical, high, medium, low)
- category (optional array: memory_leak, slow_query, api_timeout, js_error, crash, security)

**Behaviour:**

Query the incident store filtered by site, severity, and category. Enrich each incident with Claude's pattern analysis where relevant context exists.

**Returns:**

Prioritised list of bugs with severity, category, frequency, first and last seen timestamps, and suggested fix approach.

### 9.6 generate_fix

**Purpose:** Use Claude to analyse a specific incident and generate a code fix.

**Input:**
- incidentId (required)
- repositoryContext (optional: repo URL, branch, relevant file paths)

**Behaviour:**

Retrieve the full incident context including logs, stack traces, and metrics. Fetch relevant code sections if repository context is provided. Call Claude with the full context and generate a structured fix proposal including the code change, explanation, tests, and rollback plan.

**Returns:**

Fix proposal object including fix ID, changed files with diffs, explanation, test cases, rollback procedure, and confidence score.

### 9.7 apply_fix

**Purpose:** Apply a generated fix to the target repository.

**Input:**
- fixId (required)
- mode (optional: dry_run, branch_pr, direct_deploy)
- targetBranch (optional)

**Behaviour:**

Retrieve the fix proposal. Run the fix in an isolated Docker test environment. If tests pass, apply the fix according to the selected mode. In branch_pr mode, create a pull request. In direct_deploy mode, commit to the target branch and trigger the deployment hook. Log all actions and outcomes.

**Returns:**

Deployment result object including status, test outcome, PR or commit URL if applicable, and rollback instructions.

### 9.8 rollback_fix

**Purpose:** Rollback a previously applied fix.

**Input:**
- fixId (required)
- reason (optional explanation)

**Behaviour:**

Retrieve the rollback plan associated with the fix. Execute the rollback steps. Log the rollback with reason, timestamp, and actor.

**Returns:**

Rollback result object with status and confirmation.

### 9.9 get_incident_detail

**Purpose:** Retrieve the full detail of a specific incident.

**Input:**
- incidentId (required)

**Returns:**

Full incident object including timeline, logs, metrics at time of incident, related incidents, applied fixes, and current resolution status.

### 9.10 get_health_report

**Purpose:** Generate a comprehensive health report for a site across a given period.

**Input:**
- siteId (required)
- periodDays (optional, default 7)
- format (optional: summary, detailed)

**Returns:**

Report object including availability percentage, performance trends, incident count, fix success rate, and prioritised recommendations.

### 9.11 configure_alerts

**Purpose:** Configure alert rules and notification channels for a monitored site.

**Input:**
- siteId (required)
- rules (array of alert rule objects)
- channels (array of channel configurations)

**Returns:**

Saved alert configuration confirmation.

### 9.12 stop_monitoring

**Purpose:** Stop monitoring a site and optionally archive its data.

**Input:**
- siteId (required)
- archiveData (optional boolean, default true)

**Returns:**

Confirmation of monitoring stop and archive status.

## 10. MCP Resources Specification

### 10.1 metrics://{siteId}/performance

Returns live and historical performance metrics including response time, TTFB, LCP, FID, and CLS time-series data.

### 10.2 metrics://{siteId}/errors

Returns error log aggregation grouped by type, frequency, and severity.

### 10.3 metrics://{siteId}/resources

Returns server resource usage metrics including memory, CPU, and database connection pool where instrumented.

### 10.4 incidents://{siteId}/active

Returns all currently open incidents for the site.

### 10.5 incidents://{siteId}/history

Returns the full incident history for the site ordered by most recent.

### 10.6 fixes://{siteId}/history

Returns all generated and applied fixes for the site including their outcomes.

### 10.7 reports://{siteId}/daily

Returns the latest daily health report summary.

### 10.8 sites://all/overview

Returns a global overview of all monitored sites and their statuses.

## 11. Monitoring Engine Requirements

### 11.1 Purpose

The monitoring engine is responsible for continuous, scheduled data collection across all registered sites.

### 11.2 Required Checks

Implement the following check types:

- HTTP uptime check (response code, response time)
- Performance check using Lighthouse for Core Web Vitals
- Synthetic user flow check using Playwright for critical paths
- Error log ingestion from instrumented sites via webhook
- Resource usage check for instrumented server environments

### 11.3 Scheduling

Use BullMQ to schedule and manage monitoring jobs.

Each site should have its own recurring job based on its configured check interval.

The scheduler must handle:

- Missed checks due to temporary failures
- Backpressure during high-load periods
- Graceful shutdown without losing job state
- Dead letter queue for consistently failing checks

### 11.4 Baseline Management

On initial monitoring registration, collect and store a baseline of performance metrics.

Use the baseline for anomaly comparison. Recalculate the baseline periodically to account for deliberate performance improvements.

Do not compare against a stale baseline indefinitely.

### 11.5 Check Result Storage

Store all check results as time-series records in TimescaleDB.

Retain high-resolution data (per-check) for the most recent 30 days.

Retain hourly aggregates for 12 months.

Retain daily aggregates indefinitely.

### 11.6 Check Failure Handling

A check that fails must:

- Be retried up to 3 times before creating an incident
- Log the failure with full error context
- Not create duplicate incidents for the same ongoing condition
- Resolve the incident automatically when subsequent checks pass

## 12. Anomaly Detection Requirements

### 12.1 Purpose

Identify issues before they become user-facing problems.

### 12.2 Detection Methods

Implement the following detection methods:

**Threshold-Based Detection**

Trigger incidents when values breach configured thresholds.

Default thresholds:
- Response time exceeds 3000ms
- Error rate exceeds 5% in a 5-minute window
- Availability drops below 99% in a 1-hour window
- Memory usage exceeds 90% of available

Allow per-site threshold overrides.

**Statistical Anomaly Detection**

Trigger incidents when values deviate significantly from baseline behaviour.

Use rolling average and standard deviation over recent history.

Flag values outside three standard deviations as anomalies.

Do not alert on expected traffic patterns such as known low-traffic hours.

**Trend Detection**

Identify gradual degradation before it becomes a hard threshold breach.

Flag steady upward trends in response time, error rate, or memory consumption over a multi-hour window.

**Pattern Matching**

Recognise known error signatures and map them to known issue categories.

Maintain a catalogue of common framework errors, database connection errors, and runtime exceptions.

### 12.3 Incident Lifecycle

An incident must move through these states:

- Detected
- Investigating
- Fix Generated
- Fix Applied
- Monitoring Resolution
- Resolved
- Closed

An incident that degrades after being marked resolved must reopen automatically.

### 12.4 Deduplication

Do not create multiple incidents for the same ongoing condition.

Group related anomalies into a single incident where they share a common root cause signal.

## 13. Claude Integration Requirements

### 13.1 Purpose

Use Claude to provide intelligent root cause analysis, human-readable explanations, and code fix generation.

### 13.2 Analysis Context

When calling Claude for analysis, always include:

- Incident type and severity
- Metrics timeline showing before and after the issue appeared
- Relevant error logs and stack traces
- Code snippets from files implicated in the error where available
- Recent deployments or config changes if available
- Previous similar incidents and their resolutions

Never call Claude with insufficient context. If context is incomplete, collect more data before invoking analysis.

### 13.3 Fix Generation Prompt Design

The fix generation prompt must instruct Claude to return:

- Root cause summary
- Affected files with full diffs
- Explanation of what changed and why
- New or updated test cases covering the fix
- Rollback procedure
- Confidence score with reasoning

The prompt must specify exact JSON output format and Claude must be instructed not to include preamble or markdown wrapping in the JSON response.

### 13.4 Fix Quality Requirements

Claude-generated fixes must:

- Address the root cause, not just the symptom
- Be minimal in scope — change only what is necessary
- Include tests that would have caught the original issue
- Not introduce new dependencies unless absolutely required
- Include clear inline comments where the fix is non-obvious

### 13.5 Confidence Scoring

Use Claude's returned confidence score to determine next steps.

- High confidence: Proceed to automated testing and deployment based on automation level
- Medium confidence: Flag for human review before applying
- Low confidence: Present as suggestion only and do not apply automatically

### 13.6 API Cost Management

Batch related analysis requests where possible.

Cache Claude analysis results for incidents that have not changed state.

Do not call Claude repeatedly for the same unresolved incident without new information.

Rate-limit the number of Claude calls per site per hour during high-incident periods.

## 14. Safety Layer Requirements

### 14.1 Purpose

Every fix must be verified in isolation before it is applied to any environment beyond the test container.

### 14.2 Test Environment

Use Docker to create isolated test environments for each fix.

The test container must:

- Mirror the target application's runtime environment
- Install dependencies cleanly without using cached layers from previous runs
- Run the application's existing test suite
- Run any new tests generated as part of the fix
- Destroy itself cleanly after the test run completes

### 14.3 Fix Acceptance Criteria

A fix is only eligible for deployment if:

- The application starts successfully in the test container
- All pre-existing tests pass
- All new tests generated with the fix pass
- No new errors appear in the container logs during a smoke run

If any of these conditions fail, the fix must be rejected and the incident must be escalated for manual review.

### 14.4 Rollback Requirement

Every applied fix must have a rollback plan captured before the fix is deployed.

The rollback plan must include:

- The exact git revert command or revert diff
- The expected post-rollback state
- Confirmation that the rollback itself is tested

### 14.5 Post-Deployment Monitoring

After a fix is deployed, monitor the site's key metrics closely for a minimum of 30 minutes.

If metrics degrade during this window, trigger the rollback automatically regardless of automation level.

Log the degradation signal that triggered the rollback.

## 15. Automation Level Requirements

The system must support four automation levels configurable per site.

### 15.1 Alert Only

Collect metrics, detect anomalies, create incidents, and send notifications.

Do not invoke Claude or generate fixes automatically.

Suitable for: High-risk production environments where all changes require human review.

### 15.2 Suggest

Collect metrics, detect anomalies, create incidents, invoke Claude analysis, and generate fix proposals.

Present fixes as suggestions. Do not apply anything without explicit approval.

Suitable for: Most production environments where developers want AI-assisted debugging but retain full control.

### 15.3 Auto Safe

Apply fixes automatically that meet all of the following:

- Confidence score is high
- Fix touches only well-understood categories such as missing indexes, cache headers, simple null guards
- Tests pass cleanly
- No schema or infrastructure changes involved

Escalate everything else to suggest mode.

Suitable for: Mature applications with good test coverage where common safe fixes should be hands-off.

### 15.4 Auto Full

Apply all validated fixes automatically regardless of category, provided tests pass and confidence is not low.

Trigger rollback automatically on degradation.

Send a post-deploy summary notification for every applied fix.

Suitable for: Well-tested staging environments, canary deployments, or highly trusted automated pipelines.

## 16. Database Schema Requirements

### 16.1 Sites

Store site configuration and current status.

Required fields:

- id
- name
- url
- automation_level
- check_interval_seconds
- created_at
- updated_at
- last_checked_at
- current_status

### 16.2 Metrics (Time-Series)

Store all monitoring check results as time-series records using TimescaleDB hypertables.

Required fields:

- time
- site_id
- check_type
- response_time_ms
- status_code
- error_rate
- lcp
- fid
- cls
- ttfb
- availability_pct

### 16.3 Incidents

Store all detected incidents and their lifecycle state.

Required fields:

- id
- site_id
- title
- description
- category
- severity
- status
- first_detected_at
- last_seen_at
- resolved_at
- resolution_note

### 16.4 Fix Proposals

Store all generated fix proposals.

Required fields:

- id
- incident_id
- site_id
- status
- diff_json
- explanation
- test_cases_json
- rollback_procedure
- confidence_score
- generated_at
- applied_at
- applied_by
- test_result
- rollback_triggered_at
- rollback_reason

### 16.5 Alert Rules

Store configurable alert thresholds and rules per site.

Required fields:

- id
- site_id
- metric
- operator
- threshold
- window_seconds
- severity
- enabled

### 16.6 Notification Channels

Store configured notification channels per site.

Required fields:

- id
- site_id
- channel_type
- configuration_json
- enabled

## 17. Git Integration Requirements

### 17.1 GitHub Support

Use Octokit for all GitHub operations.

Required operations:

- Read file contents from a repository
- Create a new branch
- Commit changes to a branch
- Create a pull request
- Merge a pull request (for auto_full mode)
- Read recent commit history

### 17.2 Credentials

GitHub tokens must be stored as encrypted environment variables.

Never commit tokens or log them.

Support organisation-level tokens for multi-repository setups.

### 17.3 PR Format

Generated pull requests must include:

- Descriptive title referencing the incident ID
- Body explaining the root cause, the fix applied, and the test results
- Link back to the incident in the WebGuard dashboard or report
- Applied labels: automated-fix, webguard

### 17.4 GitLab Support Readiness

The git integration layer must be abstracted behind a provider interface so GitLab support can be added later without significant refactoring.

Do not couple Octokit calls directly to business logic.

## 18. Notification Requirements

### 18.1 Supported Channels

Implement notification dispatch for:

- Slack (via incoming webhooks)
- Discord (via incoming webhooks)
- Email (via SMTP or transactional email service)
- Generic webhook (POST JSON to a configured URL)

### 18.2 Notification Events

Send notifications for:

- New incident detected
- Incident severity upgraded
- Fix generated and ready for review (suggest mode)
- Fix applied successfully
- Fix failed tests
- Rollback triggered
- Site health report summary
- Monitoring resumed after outage

### 18.3 Notification Content

Each notification must include:

- Site name and URL
- Event type
- Short human-readable summary
- Severity level
- Link to full incident or fix detail

Notifications must not include raw stack traces or sensitive credential data.

### 18.4 Rate Limiting

Do not send duplicate notifications for the same ongoing incident within the same notification window.

Group rapid sequential anomalies into a single notification where appropriate.

## 19. Dashboard API Requirements

The system must expose a REST API for external dashboard consumption.

### 19.1 Required Endpoints

```
GET  /api/sites
POST /api/sites
GET  /api/sites/:siteId
PUT  /api/sites/:siteId
DELETE /api/sites/:siteId

GET  /api/sites/:siteId/health
GET  /api/sites/:siteId/metrics
GET  /api/sites/:siteId/incidents
GET  /api/sites/:siteId/incidents/:incidentId
GET  /api/sites/:siteId/fixes
GET  /api/sites/:siteId/fixes/:fixId
POST /api/sites/:siteId/fixes/:fixId/apply
POST /api/sites/:siteId/fixes/:fixId/rollback

GET  /api/reports/:siteId/daily
GET  /api/reports/:siteId/weekly

POST /api/webhooks/error-ingest/:siteId
```

### 19.2 API Principles

- Validate all inputs with Zod schemas
- Return consistent JSON response shapes
- Use HTTP status codes correctly
- Return human-safe error messages
- Authenticate all non-webhook endpoints
- Rate-limit the error ingest webhook
- Never return raw database errors or stack traces to API consumers

### 19.3 Response Shape

All successful responses must use a consistent envelope:

```json
{
  "success": true,
  "data": { ... }
}
```

All error responses must use:

```json
{
  "success": false,
  "error": {
    "code": "INCIDENT_NOT_FOUND",
    "message": "The requested incident could not be found."
  }
}
```

## 20. Logging Requirements

### 20.1 Structured Logging

Use Pino for all logging.

Log entries must be structured JSON with:

- timestamp
- level
- service
- traceId (for request-scoped logs)
- message
- relevant contextual fields

### 20.2 Log Levels

Use log levels appropriately:

- error: Unrecoverable failures requiring attention
- warn: Recoverable issues or unexpected conditions
- info: Normal operational events
- debug: Detailed diagnostic information (disabled in production by default)

### 20.3 Never Log

- API keys or secrets
- Raw authentication tokens
- Full request bodies containing credentials
- Personally identifiable information

### 20.4 Audit Trail

All automated actions must be logged with:

- Actor (system or user identifier)
- Action type
- Target resource
- Outcome
- Timestamp

This audit trail must be stored in the database and must not be deletable through normal API operations.

## 21. Validation Requirements

Use Zod for all input validation.

Validate:

- All MCP tool inputs
- All REST API request bodies and query parameters
- All webhook payloads
- All environment variable configuration on startup
- All external API responses before parsing them into internal types

If environment variable validation fails on startup, the server must exit with a clear error message listing the missing or invalid variables.

Do not silently swallow validation errors. Return structured validation error responses.

## 22. Error Handling Requirements

### 22.1 MCP Tool Errors

All MCP tool handlers must catch errors and return structured MCP error objects.

Do not allow unhandled exceptions to crash the server process.

Log full error context at the server level while returning a clean message to the MCP client.

### 22.2 External Failure Isolation

Failures in Playwright checks, Lighthouse audits, or Claude API calls must not crash the monitoring engine.

Each check type must have independent error boundaries.

Log the failure, increment the failure counter, and continue the monitoring loop.

### 22.3 Database Errors

Wrap all database operations in try/catch blocks.

Log database errors with the query context for debugging.

Do not expose raw database error messages to API consumers or MCP clients.

### 22.4 Queue Failures

BullMQ jobs that fail must be retried according to the configured retry policy.

Jobs that exhaust retries must be moved to the dead letter queue.

The dead letter queue must be monitored and alert if it grows beyond a configured threshold.

## 23. Performance Requirements

The system must remain responsive under load.

Implement:

- Connection pooling for PostgreSQL
- Redis caching for frequently accessed site configurations and recent metrics
- Batch metric writes to the database rather than individual row inserts
- Debounced duplicate incident creation
- Pagination for all list endpoints
- Streaming responses for large report downloads where applicable
- Non-blocking monitoring job execution

The MCP server response time for non-analysis tools must stay below 500ms under normal conditions.

Analysis tools that invoke Claude may take longer and must communicate their working state appropriately.

## 24. Security Requirements

Implement the system with good security hygiene.

Required:

- Environment variable secrets only
- Encrypted storage for third-party tokens at rest
- API authentication for the dashboard REST API
- Webhook signature verification for the error ingest endpoint
- Input validation on all entry points
- Rate limiting on public-facing endpoints
- No raw secret logging
- No sensitive data in MCP tool responses beyond what is necessary
- Secure Docker image practices for test containers
- Dependency audit as part of CI pipeline

## 25. Testing Requirements

Use Vitest for unit and integration testing.

### 25.1 Required Test Coverage

At minimum, verify:

- MCP tool registration and schema validation
- Monitoring check execution and result storage
- Anomaly detection logic for all detection methods
- Incident creation and deduplication
- Fix generation pipeline from incident to diff
- Safety layer container test execution
- Fix application and rollback logic
- Alert rule evaluation
- Notification dispatch
- Dashboard API endpoint behaviour
- Error ingest webhook
- Authentication and protected routes
- Queue scheduling and retry logic

### 25.2 Integration Tests

Write integration tests that cover the end-to-end flow:

- Register site → run check → detect anomaly → create incident
- Create incident → generate fix → run tests → apply fix → validate resolution

Use a dedicated test database and Docker-in-Docker for container tests in CI.

### 25.3 Test Isolation

Tests must not share state.

Each test must set up and tear down its own data.

Do not rely on external network calls in unit tests. Mock all external dependencies.

## 26. Deployment Requirements

### 26.1 Container Configuration

The entire system must be deployable using Docker Compose for local and staging environments.

Required services in Docker Compose:

- webguard-mcp (the MCP server)
- postgres (with TimescaleDB)
- redis
- worker (BullMQ worker process)

### 26.2 Environment Configuration

All configuration must be loaded from environment variables.

Provide a complete .env.example file documenting every required variable.

Group variables by category:

- Database
- Redis
- Anthropic API
- GitHub integration
- Notification channels
- Monitoring defaults
- Server settings

### 26.3 CI/CD Pipeline

Implement a GitHub Actions pipeline that:

- Runs linting and type checking
- Runs the full test suite
- Builds the Docker image
- Runs a container-level smoke test
- Deploys to the target environment on main branch merge

### 26.4 Health Endpoint

Expose a GET /health endpoint that returns the current server status, database connectivity, Redis connectivity, and queue worker status.

This endpoint must not require authentication.

## 27. Implementation Priority

Build in this order:

1. Project setup with selected stack, TypeScript config, Drizzle, Docker Compose
2. Environment validation and configuration loading
3. Database schema and migrations
4. Redis and BullMQ setup
5. Core MCP server with stdio transport
6. HTTP/SSE transport layer
7. Monitoring engine: HTTP uptime check
8. Monitoring engine: Lighthouse performance check
9. Monitoring engine: Playwright synthetic check
10. Time-series metrics storage
11. Threshold-based anomaly detection
12. Statistical anomaly detection
13. Incident lifecycle management
14. Claude integration: root cause analysis
15. Claude integration: fix generation
16. Safety layer: Docker test runner
17. Git integration: read code, create branch, PR
18. Fix application and rollback pipeline
19. Alert rule evaluation
20. Notification dispatch
21. Dashboard REST API
22. Error ingest webhook
23. MCP resources
24. All remaining MCP tools
25. Report generation
26. Multi-site support validation
27. Audit logging
28. Test suite completion
29. CI/CD pipeline
30. Final end-to-end validation

## 28. Features Explicitly Not in Scope

Do not implement these:

- Billing or subscription management
- Public user accounts or self-service registration
- Social or community features
- AI model fine-tuning or custom training
- Full browser DevTools protocol replacement
- Streaming video or real user monitoring agent injection
- Native mobile app
- Full APM suite replacing Datadog or New Relic
- Admin UI dashboard (the REST API is sufficient)
- Multi-cloud infrastructure provisioning
- Kubernetes operator

Avoid adding these even if they seem like natural extensions. Keep the implementation focused.

## 29. Final Acceptance Criteria

The implementation is successful when:

1. A developer can register a website and have monitoring begin within 60 seconds.
2. A performance regression is detected within two check intervals of it occurring.
3. An incident is created automatically without human action.
4. Claude generates a structured, reasonable fix proposal for a detected memory leak or slow query scenario.
5. The fix is tested in an isolated Docker container and the test result is accurate.
6. In suggest mode, the fix is presented as a pull request without being deployed.
7. In auto_safe mode, a high-confidence safe fix is applied and the post-deploy monitoring begins automatically.
8. A degradation signal after deployment triggers a rollback automatically.
9. All monitoring, incidents, fixes, and audit events are accessible through MCP tools and resources.
10. The MCP server integrates cleanly with Claude.ai and responds to natural language queries about site health.
11. The system handles external API failures, queue failures, and database errors without crashing.
12. All secrets are loaded from environment variables and never appear in logs or responses.

If this loop works reliably, the implementation is complete.

## 30. Final Instruction to Dev Agent

Build WebGuard MCP as a full, production-ready, autonomous website monitoring and remediation server using the stack direction provided.

Use the latest official documentation for every library and tool. Make smart implementation decisions where exact technical choices are not specified. Do not waste time on unnecessary features. Focus on completing the monitoring and remediation loop with reliable automation, strong code quality, secure credential handling, and clean MCP tool integration.

Every automated action must be traceable. Every deployed fix must be reversible. Every error must be handled gracefully.

The final result should be a system that a real engineering team can point at a production website and trust.