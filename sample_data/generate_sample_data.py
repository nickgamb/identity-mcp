#!/usr/bin/env python3
"""Generate sample conversation and memory data for identity-mcp.

Creates a fictional persona (Morgan Riley — platform engineer, sourdough baker,
home-automation tinkerer) with a distinctive and consistent writing style so that
the identity model training pipeline has enough signal to build a real fingerprint.

Output:
    sample_data/conversations/   — ~25 JSONL conversation files (~400 user messages)
    sample_data/memory/memories.json — ChatGPT-style memory export (~80 entries)

Usage:
    python generate_sample_data.py          # writes into ./conversations and ./memory
    python generate_sample_data.py --out /path/to/output
"""

import json
import os
import uuid
import argparse
from datetime import datetime, timedelta

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Persona: Morgan Riley
# ---------------------------------------------------------------------------
# Platform engineer at a fintech company. Writes Go and Python. Builds
# event-driven microservices. Side interests: sourdough baking, home
# automation (Home Assistant / ESPHome), and trail running.
#
# Style markers the identity model should pick up:
#   - Uses em-dashes frequently
#   - Starts thoughts with "the thing is" or "fundamentally"
#   - Tends toward systems thinking even in non-tech contexts
#   - Concise but precise — avoids filler
#   - Distinctive vocabulary: backpressure, circuit breaker, saga pattern,
#     idempotency, observability, trace propagation, hydration percentage,
#     crumb structure, autolyse, bulk ferment, zigbee, mqtt
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    # --- Technical conversations ---
    {
        "topic": "circuit-breaker-design",
        "messages": [
            ("user", "I need to add circuit breaker logic to our payment gateway client. Right now if the downstream provider goes down we just hammer it with retries until the timeout cascade takes out the whole service mesh."),
            ("assistant", "A circuit breaker pattern would help here. You'd track failure rates over a sliding window and trip the breaker when failures exceed a threshold, returning a fast error instead of waiting for timeouts. Would you like me to outline the state machine (closed → open → half-open)?"),
            ("user", "Yeah — the thing is, we already have a half-baked retry wrapper but it doesn't track state across requests. I want something that integrates with our Prometheus metrics so we get observability on trip events. Fundamentally the circuit breaker needs to be per-endpoint, not per-service, because the provider has separate endpoints for auth and capture that fail independently."),
            ("assistant", "That's a good distinction. Per-endpoint breakers give you much finer control. For the Prometheus integration, you'd want counters for trips, resets, and half-open probe results. Here's a Go pattern using a struct per endpoint..."),
            ("user", "Right, and I want the thresholds configurable via our feature flags system — not hardcoded. The auth endpoint can tolerate maybe 30% failure rate before tripping because it's called way more often, but the capture endpoint should trip at 10% because every failure there is a lost transaction."),
            ("assistant", "Makes sense. You could define a CircuitBreakerConfig struct with fields for failure threshold, window duration, and half-open max probes, then load those from your feature flag system at startup and on change events."),
            ("user", "Exactly. And the half-open state needs backpressure — I don't want to flood the recovering endpoint with probe requests. One probe per interval, and if it succeeds we ramp back up gradually. Think exponential ramp-up, not instant close."),
            ("assistant", "So a graduated close: half-open allows 1 request, then 2, 4, 8 up to full traffic. Each successful batch doubles the allowed concurrency. A single failure at any ramp stage re-trips to open."),
            ("user", "That's the pattern. I'll prototype it tomorrow — going to use our existing middleware chain so it composes cleanly with the trace propagation and idempotency layers we already have."),
        ],
    },
    {
        "topic": "event-sourcing-migration",
        "messages": [
            ("user", "We're thinking about migrating the ledger service from CRUD to event sourcing. The current Postgres schema is getting unwieldy — 47 columns on the transactions table and we keep adding more for audit requirements."),
            ("assistant", "Event sourcing would give you a complete audit trail by design. Instead of mutating rows, you'd append immutable events and derive the current state by replaying them. The tradeoff is query complexity — you'd need projections for read-side queries."),
            ("user", "The thing is, we already have a pseudo-event-log table that the compliance team made us add. Every mutation writes a row to transaction_audit with the before/after JSON. So we're paying the write cost of event sourcing without the benefits — worst of both worlds."),
            ("assistant", "That's a common pattern I see. You're essentially maintaining two sources of truth. With proper event sourcing, the events *are* the truth and the materialized view replaces the transactions table."),
            ("user", "Fundamentally that's the pitch I'm making to the team. My concern is the replay performance — we have 180M events in that audit table already. Cold replay from zero would take hours. I'm thinking we need periodic snapshots so replay only covers the delta."),
            ("assistant", "Snapshots are standard practice for event-sourced systems at that scale. You'd take a snapshot of the aggregate state every N events (say 1000) and replay only from the latest snapshot. That bounds your replay time regardless of total event count."),
            ("user", "Right. And for the read side I want to use ClickHouse — we already have it for analytics and it handles the kind of time-series queries compliance wants. The projection from the event store to ClickHouse would go through our Kafka topic."),
            ("assistant", "That's a clean architecture: events → Kafka → ClickHouse projections for reads, plus snapshots in the event store for write-side replay. You'd want idempotent consumers on the Kafka side to handle redelivery."),
            ("user", "Yeah, idempotency keys on every event. We learned that lesson the hard way last quarter when a Kafka consumer restart caused duplicate settlement events. Cost us three days of reconciliation."),
            ("assistant", "Painful but common. With event sourcing, each event has a natural sequence number within its aggregate, which gives you built-in idempotency for replay. The Kafka consumer just needs to track its last processed sequence per aggregate."),
            ("user", "Good. I'm going to write up an RFC this week. The saga pattern for the multi-step settlement flow is the part that worries me most — it's currently a stored procedure that does six things in one transaction."),
        ],
    },
    {
        "topic": "kubernetes-debugging",
        "messages": [
            ("user", "Our staging cluster is doing something weird — pods keep getting evicted from one specific node but the node metrics look fine. Memory and CPU both under 60%. I've been staring at kubectl describe node output for an hour."),
            ("assistant", "Check the ephemeral storage. Kubernetes will evict pods when the node's ephemeral storage usage exceeds the eviction threshold, even if CPU and memory are fine. Run `kubectl describe node <name>` and look for the `ephemeral-storage` condition."),
            ("user", "Oh — that's it. The node shows ephemeral-storage pressure. But we don't write much to disk... wait, it's the container logs. Our payment service is writing debug-level logs at like 500MB per hour because someone left verbose logging on in the staging config."),
            ("assistant", "Classic. The kubelet counts container log files toward ephemeral storage. You can either fix the log level or set resource limits for ephemeral storage on the pod spec so it gets evicted individually instead of causing node-level pressure."),
            ("user", "I'll fix the log level — that's the real problem. But fundamentally we should have caught this earlier. I'm going to add a Grafana alert on node ephemeral storage utilization. We have alerts for CPU and memory but never added one for disk."),
            ("assistant", "Good call. The node_filesystem_avail_bytes metric from node-exporter gives you what you need. Alert when any node drops below, say, 15% available ephemeral storage."),
            ("user", "Yeah. I'll add it to our Argo CD managed alerting rules so it deploys with the monitoring stack. The thing is — we keep discovering these gaps reactively. I want to do a full observability audit next sprint."),
        ],
    },
    {
        "topic": "grpc-service-design",
        "messages": [
            ("user", "Starting a new service for the risk scoring engine. Debating between REST and gRPC — the service will be called synchronously in the payment authorization path so latency matters a lot."),
            ("assistant", "For synchronous, latency-sensitive internal service communication, gRPC has clear advantages: binary protobuf serialization is faster than JSON, HTTP/2 multiplexing reduces connection overhead, and streaming support could be useful if you need to score multiple transactions in a batch."),
            ("user", "That's my thinking. The protobuf schema also gives us a versioned contract which is better than our current situation where REST endpoints drift between what the docs say and what the code does. I'm tired of debugging integration issues caused by undocumented field changes."),
            ("assistant", "Proto files as the single source of truth for service contracts is one of gRPC's biggest wins. You can generate client libraries in Go, Python, and TypeScript from the same .proto file."),
            ("user", "Exactly. And we can use buf for linting and breaking change detection in CI. The thing is, I also need to expose this to our internal dashboard which is a React app — gRPC-Web or a gateway?"),
            ("assistant", "gRPC-Web through Envoy is the cleaner path. You'd add a gRPC-Web filter to the Envoy config in your service mesh, which translates browser-compatible requests to native gRPC. No separate REST API to maintain."),
            ("user", "Good — we already have Envoy in the mesh. I'll prototype the proto schema today and get it reviewed before writing any service code. Schema-first is the way."),
        ],
    },
    {
        "topic": "temporal-workflow-engine",
        "messages": [
            ("user", "I've been evaluating Temporal for our long-running business processes. Right now we have this Rube Goldberg machine of cron jobs, SQS queues, and DynamoDB state tables that implement what's basically a workflow engine — badly."),
            ("assistant", "Temporal would replace all of that with explicit workflow definitions. Your business logic becomes a function with retries, timeouts, and state management handled by the platform. What workflows are you looking at migrating first?"),
            ("user", "The merchant onboarding flow — it's a 14-step process that takes 3-7 days, involves external KYC checks, manual review steps, and has branching logic depending on the merchant's risk tier. Right now if any step fails we have a dead letter queue and someone manually restarts from a spreadsheet."),
            ("assistant", "That's a perfect Temporal use case. The workflow function would model each step, with activities for the external calls and signals for the manual review approvals. Temporal handles retry, timeout, and state persistence automatically."),
            ("user", "The manual review step is interesting — I can use Temporal signals for that. When the compliance team approves in our admin UI, it sends a signal to the workflow which unblocks and continues. No polling, no callback hell."),
            ("assistant", "Exactly. Signals are the idiomatic way to handle human-in-the-loop steps. The workflow function just awaits the signal, and Temporal persists the state while waiting — could be minutes or days."),
            ("user", "Fundamentally this is why I want Temporal — it makes the invisible state visible. Right now nobody can tell you where a merchant onboarding is stuck without checking five different systems. With Temporal I can query the workflow state directly."),
            ("assistant", "And Temporal's UI gives you visibility into all running workflows, their current state, and history. Your ops team can see exactly where each merchant onboarding is without any custom dashboards."),
            ("user", "I'm going to run a proof of concept this sprint. One concern — our Go services use a specific error handling pattern with wrapped errors and I need to make sure that plays well with Temporal's activity retry logic."),
            ("assistant", "Temporal activities can return application-level errors that control retry behavior. You can mark errors as non-retryable to prevent infinite retries on permanent failures like 'merchant already exists.'"),
            ("user", "Right, that's critical. The saga pattern comes in too — if step 8 fails after steps 1-7 succeeded, some of those need compensating transactions. Temporal's built-in compensation support should handle that."),
        ],
    },
    {
        "topic": "database-connection-pooling",
        "messages": [
            ("user", "We're hitting Postgres connection limits during peak traffic. Currently at max_connections=200 and our 12 service replicas are each configured with a pool of 20 connections. The math doesn't work — 12 * 20 = 240 > 200."),
            ("assistant", "That's the root cause. You're oversubscribing the database connection limit. Two approaches: increase max_connections (has memory implications) or use PgBouncer as a connection pooler between your services and Postgres."),
            ("user", "PgBouncer is the right call — I don't want to keep bumping max_connections every time we scale. The thing is, we use prepared statements heavily and I've heard PgBouncer in transaction mode doesn't support them."),
            ("assistant", "That's correct for older PgBouncer versions, but PgBouncer 1.21+ supports prepared statements in transaction pooling mode. Make sure you're on a recent version."),
            ("user", "Good to know. I'll set it up with transaction pooling — each service gets a logical pool of 20 but PgBouncer multiplexes onto a smaller set of actual connections. Fundamentally this is the same pattern as our HTTP connection pooling, just at the database layer."),
            ("assistant", "Exactly. PgBouncer acts as a connection concentrator. You could set it to maintain 100 server connections to Postgres and let all 12 services share that pool. Most of the time connections are idle between query executions anyway."),
            ("user", "Yeah, and with transaction pooling the connection is only held during the transaction, not the entire session. Our transactions are short — sub-millisecond for most reads. I'll deploy PgBouncer as a sidecar in the pod."),
        ],
    },
    {
        "topic": "observability-tracing",
        "messages": [
            ("user", "I want to add distributed tracing across our service mesh. We have Prometheus for metrics and Loki for logs but no trace propagation — when something goes wrong we're stitching together logs from six services by timestamp and request ID."),
            ("assistant", "OpenTelemetry would be the standard choice. You'd instrument each service with the OTEL SDK, propagate trace context via HTTP headers (W3C traceparent), and export spans to a collector that stores them in Jaeger or Tempo."),
            ("user", "We already have Grafana so Tempo makes sense — keeps everything in one stack. The thing is, I want trace context to flow through Kafka too, not just synchronous HTTP calls. When service A publishes an event and service B consumes it, I want those to show up as part of the same trace."),
            ("assistant", "You'd inject the trace context into the Kafka message headers when producing, then extract it when consuming to continue the trace. The OTEL Kafka instrumentation libraries handle this for both Go and Python."),
            ("user", "Perfect. And I want to correlate traces with our Prometheus metrics — exemplars, right? So when I see a latency spike in Grafana I can click through to the specific traces that caused it."),
            ("assistant", "Exactly. Prometheus exemplars attach trace IDs to metric observations. When you view a histogram in Grafana, exemplar dots appear that link directly to Tempo traces. It's the missing link between metrics and traces."),
            ("user", "Fundamentally the observability stack should tell a story — metrics say something is wrong, traces show where, logs explain why. We have two out of three and traces are the gap. I'm going to prioritize this for the next quarter."),
        ],
    },
    # --- Sourdough / baking conversations ---
    {
        "topic": "sourdough-hydration",
        "messages": [
            ("user", "I've been struggling with my sourdough crumb — it's consistently dense in the center even though the crust is great. Running 75% hydration with bread flour."),
            ("assistant", "A dense center usually points to either underfermentation or shaping issues. How long is your bulk ferment and what's your ambient temperature?"),
            ("user", "Bulk ferment is about 5 hours at around 76°F. I do four sets of stretch and folds every 30 minutes in the first two hours, then leave it. The thing is, the dough passes the window pane test and looks well-developed — it's not a gluten issue."),
            ("assistant", "If gluten development looks good, I'd suspect the bulk isn't going long enough. At 76°F you might need closer to 6-7 hours. Are you seeing a 50% volume increase by the end of bulk?"),
            ("user", "Hmm, probably more like 30-40%. I've been going by time rather than dough feel. Fundamentally I should be reading the dough, not the clock — same principle as monitoring systems, the metrics tell you the truth."),
            ("assistant", "That's a great analogy. The dough should increase 50-75% in volume, feel jiggly and aerated, and show visible bubbles on the surface and sides. Time is just a rough guide that varies with temperature and starter activity."),
            ("user", "I'm also wondering about my autolyse. I'm only doing 30 minutes but I've read that a longer autolyse — like 2 hours — can improve the crumb structure by giving the enzymes more time to break down the starches."),
            ("assistant", "A longer autolyse does help with extensibility and can improve crumb openness. Just make sure you're doing a flour-and-water-only autolyse without the levain, since the levain's acids can tighten the gluten during a long autolyse."),
            ("user", "Right, flour and water only, then mix in the levain and salt after. I'll try a 2-hour autolyse and extend the bulk until I see that 50% rise. Baking is just another system to debug — change one variable at a time."),
        ],
    },
    {
        "topic": "sourdough-starter-maintenance",
        "messages": [
            ("user", "My starter has been sluggish this week — taking 10 hours to peak instead of the usual 5-6. I haven't changed my feeding ratio or flour. Could be the temperature drop — my kitchen is running about 66°F now that fall hit."),
            ("assistant", "Temperature has a huge impact on fermentation rate. At 66°F your starter's microbial activity slows significantly. You could try feeding with warmer water (85-90°F) to give it a boost, or find a warmer spot like on top of your fridge."),
            ("user", "I actually built a proofing box with a seedling heat mat and an Inkbird temperature controller — same one I use for the fermentation chamber. Set it to 78°F and the starter should be back on schedule."),
            ("assistant", "That's a solid setup. A consistent 78°F is ideal for starter activity. The Inkbird controller keeps it precise, which removes one more variable from your baking process."),
            ("user", "Exactly — the thing is, baking and engineering have the same core principle: control your variables and the outcomes become predictable. My proofing box is basically a PID controller for fermentation."),
            ("assistant", "The parallel is surprisingly strong. Both are about understanding a system well enough to control its inputs and predict its outputs."),
            ("user", "Ha, yeah. My wife thinks I'm overthinking bread but the crumb doesn't lie. Anyway, I also want to try a rye flour addition to the starter — maybe 10% rye — since rye has more wild yeast and minerals. Should boost the fermentation activity."),
        ],
    },
    {
        "topic": "baking-schedule-optimization",
        "messages": [
            ("user", "I need to figure out a baking schedule that works with my work schedule. Right now I'm doing the bulk ferment during the day but I keep missing the shaping window because I get pulled into meetings."),
            ("assistant", "A cold retard approach would give you more flexibility. You could mix and do your folds in the evening, let it bulk ferment overnight in the fridge, then shape and bake in the morning before work."),
            ("user", "That's what I've been thinking. The fridge slows fermentation enough that I get an 8-12 hour window instead of needing to catch it at exactly the right moment. Same concept as message queues — buffer the work and process it when you're ready."),
            ("assistant", "The cold retard also develops more complex flavors from the slower fermentation. Many bakers prefer it specifically for the flavor profile."),
            ("user", "Right, the lactic acid bacteria are more active at lower temperatures relative to the yeast, so you get more tang. I like that. My plan is: mix at 8pm, folds until 10pm, into the fridge overnight, pull at 6am, shape, proof 2 hours, bake at 8am before standup."),
            ("assistant", "That schedule works well. The 2-hour final proof at room temperature after cold retard should be enough for a good oven spring, especially if the bulk went well."),
            ("user", "Perfect. Fundamentally this is about designing the process around my constraints rather than fighting them — same thing I tell my team about system design."),
        ],
    },
    # --- Home automation conversations ---
    {
        "topic": "home-assistant-zigbee",
        "messages": [
            ("user", "I'm migrating my smart home from a mix of WiFi devices to Zigbee. The WiFi stuff is unreliable — the smart plugs drop off the network every few days and I have to power cycle them. Plus they're all phoning home to different cloud services."),
            ("assistant", "Zigbee is much more reliable for home automation. It forms a mesh network where devices route through each other, so the network self-heals. And with a local coordinator like ZHA or Zigbee2MQTT in Home Assistant, everything stays local — no cloud dependency."),
            ("user", "I'm going with Zigbee2MQTT because it gives me more control over the device pairing and configuration. Running it on the same Raspberry Pi as Home Assistant with a Sonoff Zigbee coordinator. The thing is, I want to keep my existing MQTT broker since I already have ESPHome sensors publishing to it."),
            ("assistant", "Zigbee2MQTT publishes device states to MQTT topics, so it integrates perfectly with your existing broker. Your ESPHome sensors and Zigbee devices would all be on the same MQTT bus, which simplifies your automations."),
            ("user", "Exactly — unified message bus, single source of truth for device state. I have about 30 ESPHome sensors already — temperature, humidity, motion, door contacts. Adding Zigbee gives me the smart plugs and light switches without the cloud dependency."),
            ("assistant", "That's a solid architecture. ESPHome for custom sensors where you need specific hardware, Zigbee for commercial devices. All communicating over MQTT with Home Assistant as the orchestration layer."),
            ("user", "And the Zigbee mesh means I don't need to worry about WiFi range for the smart plugs in the garage and basement. Every mains-powered Zigbee device acts as a router, so the mesh extends itself. Fundamentally it's the same topology as a service mesh — each node routes for others."),
            ("assistant", "The analogy works well. Like Envoy sidecars routing traffic, Zigbee routers forward messages for devices that can't reach the coordinator directly. Battery-powered devices (end devices) are like lightweight clients that connect through the nearest router."),
            ("user", "Ha, I'm going to start calling my smart plugs 'sidecar proxies' and see if my wife notices. Anyway — I need to figure out the MQTT topic structure for the Zigbee devices so they're consistent with my ESPHome naming convention."),
        ],
    },
    {
        "topic": "esphome-sensor-project",
        "messages": [
            ("user", "Building a new ESPHome sensor node for the garage — I want temperature, humidity, and a PIR motion sensor all on one ESP32 board. The garage isn't heated so it needs to handle temperature swings from -10°F to 110°F."),
            ("assistant", "The ESP32 itself handles that range fine. For temperature and humidity, a BME280 would be a good choice — it's accurate and handles the full range. For the PIR, an AM312 is compact and low power."),
            ("user", "Yeah, the BME280 is what I use everywhere else. I want to add a door contact sensor too — magnetic reed switch on the garage door so I can track open/close state. The ESPHome binary sensor platform handles that natively."),
            ("assistant", "A reed switch is the simplest approach. Wire it between a GPIO pin and ground, enable the internal pull-up in ESPHome, and you have a reliable open/close sensor. You could also add a template sensor that tracks how long the door has been open."),
            ("user", "Good idea on the duration tracker — I want an automation that alerts me if the garage door is open for more than 30 minutes. I'll use a Home Assistant automation for that since the logic spans multiple sensors. The ESPHome node just publishes state, the orchestration layer handles the business logic."),
            ("assistant", "Clean separation of concerns. The sensor node is responsible for data collection and publishing, Home Assistant handles the automation logic. Same as keeping your microservices focused on one responsibility."),
            ("user", "That's literally how I think about it. Each ESPHome node is a microservice — single responsibility, well-defined interface (MQTT topics), independently deployable. The thing is, I actually use the same CI/CD mindset for firmware updates — OTA updates pushed from a central repo."),
        ],
    },
    {
        "topic": "home-energy-monitoring",
        "messages": [
            ("user", "I want to build a whole-house energy monitoring system. I've been looking at CT clamps on the electrical panel — non-invasive current transformers that measure power consumption per circuit."),
            ("assistant", "CT clamps are the standard approach for DIY energy monitoring. The Emporia Vue or IoTaWatt are popular options that give per-circuit monitoring. Or you could build your own with an ESP32 and SCT-013 CT sensors."),
            ("user", "I'd rather build it myself with ESPHome so it fits my existing stack. The SCT-013 clamps output an analog signal I can read with the ESP32's ADC. The thing is, the ESP32's ADC is notoriously noisy — I might need an external ADC like the ADS1115 for accurate power readings."),
            ("assistant", "The ADS1115 is a good choice — 16-bit resolution over I2C with four channels. You'd need one per four circuits you want to monitor. ESPHome has native support for the ADS1115 as a sensor platform."),
            ("user", "Right. I have 20 circuits I care about, so that's 5 ADS1115 boards. A bit much but manageable. I'll put them on a single I2C bus — each ADS1115 has a configurable address so I can have four per bus, need two I2C buses total."),
            ("assistant", "With 5 boards you'd want addresses 0x48 through 0x4B on one bus and 0x48 on a second bus. The ESP32 supports two hardware I2C buses natively."),
            ("user", "Fundamentally this is a data pipeline problem — sensor → ADC → ESP32 → MQTT → Home Assistant → Grafana. Same architecture as any telemetry system. I want to calculate cost per circuit by multiplying power readings by our utility rate and graphing it in Grafana over time."),
        ],
    },
    # --- Mixed / personal conversations ---
    {
        "topic": "trail-running-gear",
        "messages": [
            ("user", "I signed up for a 50K trail race in June — it's in the Cascades so there's going to be significant elevation gain. I've been running 30-40 mile weeks on road but I need to shift to trail training."),
            ("assistant", "For a 50K in the Cascades you'll want to prioritize elevation training and technical trail skills over pure mileage. Time on feet matters more than distance at that point. How much elevation gain does the race have?"),
            ("user", "About 6,500 feet of gain over the 50K. The course has a mix of single track, fire roads, and one rocky ridge section. The thing is, I live in a pretty flat area so I'll need to find creative ways to train for the elevation."),
            ("assistant", "Stair repeats and treadmill incline work can partially substitute for real hills. Some runners also use a weighted vest on flat trails to simulate the effort of climbing. For the technical sections, find the rockiest trails you can access for weekend long runs."),
            ("user", "I've been doing stair repeats at the parking garage near work — 8 floors, 10 repeats. It's boring but effective. For the long runs I drive an hour to the state park that has decent single track with some roots and rocks. I'd rather optimize the training than suffer on race day."),
            ("assistant", "Smart approach. You might also want to practice hiking the steeps — at those grades and that distance, power hiking the climbs is faster and more efficient than trying to run everything."),
            ("user", "Yeah, I've been reading about the run-hike strategy for ultras. Fundamentally it's about energy management over the full distance — optimize for total time, not per-mile pace. Same as capacity planning for a distributed system — you can't run every service at 100% CPU."),
        ],
    },
    {
        "topic": "reading-recommendations",
        "messages": [
            ("user", "I just finished Designing Data-Intensive Applications by Kleppmann and it's easily the best technical book I've read in years. The chapter on stream processing completely changed how I think about our Kafka architecture."),
            ("assistant", "DDIA is a masterpiece — it gives you the mental models to reason about distributed systems trade-offs without getting lost in implementation details. Which aspects of stream processing shifted your thinking?"),
            ("user", "The idea that a log is the fundamental data structure — that tables are just materialized views of a log. Once you see it that way, the boundary between databases and message queues dissolves. Our Kafka topics are just unbounded tables, and our Postgres tables are just cached projections."),
            ("assistant", "That's the key insight from the log-centric architecture chapter. It unifies event sourcing, CQRS, CDC, and stream processing under one conceptual model."),
            ("user", "Exactly. The thing is, I wish we'd read this book as a team before we designed our current architecture. We would have made very different choices around how data flows between services. Instead we built a bunch of request-response APIs and bolted on Kafka after the fact."),
            ("assistant", "It's a common trajectory. Most teams start with synchronous REST and only add async messaging when they hit the limits. At least now you have the conceptual framework for the next evolution."),
            ("user", "Yeah. I'm going to do a book club with the team — one chapter a week. Also picked up Database Internals by Petrov as my next read. I want to understand the storage engine layer better since we're evaluating ClickHouse for our analytics workload."),
        ],
    },
    {
        "topic": "work-life-balance",
        "messages": [
            ("user", "I need to set better boundaries around on-call. Last week I got paged three times between midnight and 6am and I was completely useless the next day. The alerts were all for things that could have waited until morning."),
            ("assistant", "Sounds like your alert severity levels need tuning. Pages that wake people up should only be for customer-impacting issues that need immediate action. Everything else should be a next-business-day ticket."),
            ("user", "That's exactly right. We have alerts firing on things like 'disk usage above 70%' at 2am — that's not an emergency, that's a Tuesday morning task. The thing is, the team is afraid to downgrade alerts because we once missed a real outage that started as a disk warning."),
            ("assistant", "That's a common fear. The solution is better alert design, not more pages. A disk at 70% gets a ticket. A disk at 90% and growing gets a page. The first gives you time, the second demands action."),
            ("user", "Fundamentally we need to distinguish between signals that need human intervention right now versus signals that need human attention during business hours. I'm going to propose a two-tier on-call: page-worthy and ticket-worthy, with clear criteria for each."),
            ("assistant", "Two-tier with documented criteria is the standard approach. Google's SRE book has a good framework for this — the page should answer 'what action do I need to take right now that can't wait 8 hours?'"),
            ("user", "Yeah, and if the answer is 'nothing' then it shouldn't have woken me up. I'm also going to push for an on-call handoff meeting every Monday where the outgoing person briefs the incoming person on active issues. We keep losing context across rotations."),
        ],
    },
    {
        "topic": "mentoring-junior-dev",
        "messages": [
            ("user", "I've been mentoring a junior dev on the team and she keeps writing these 500-line functions with deeply nested if-else chains. I don't want to just rewrite her code — I want to help her develop the instinct for when to extract a function or simplify control flow."),
            ("assistant", "A good approach is to pair with her on a refactoring session. Start with one of her functions, walk through the logic together, and let her identify the natural extraction points. Asking 'what is this block responsible for?' often helps people see the function boundaries."),
            ("user", "That's what I've been doing — Socratic method. The thing is, she's smart and she gets it in the moment, but the next PR has the same pattern. I think it's a matter of practice, not understanding. She knows what clean code looks like but defaults to the first thing that works when she's under time pressure."),
            ("assistant", "That's common for junior developers. The 'first thing that works' instinct is strong. One technique is to have her write the messy version first (get it working), then do a cleanup pass before opening the PR. Making refactoring a separate, explicit step can help build the habit."),
            ("user", "Good idea. I'll suggest that workflow — green tests first, then refactor. Fundamentally I think the skill is recognizing the smell while you're writing it, not after. But that takes years of practice. For now, a two-pass approach gets the same result."),
            ("assistant", "And over time, the second pass gets faster and starts happening during the first pass. It's like how experienced musicians practice scales until the fingerings become automatic — the pattern recognition becomes subconscious."),
            ("user", "Nice analogy. I'm also going to start giving her smaller, more focused tasks. Her last ticket was 'implement the reconciliation pipeline' which is way too broad — of course she ended up with a monolithic function. Smaller scope, cleaner code."),
        ],
    },
    {
        "topic": "team-architecture-review",
        "messages": [
            ("user", "We're doing an architecture review for Q3 and I want to propose replacing our custom service discovery with Consul. We built the current system two years ago and it's become a maintenance burden — we spend more time fixing discovery than building features."),
            ("assistant", "Consul gives you service discovery, health checking, and KV storage in one package. The main consideration is the operational overhead of running the Consul cluster itself — 3 or 5 servers for the consensus group."),
            ("user", "We're on Kubernetes so I'd run it as a StatefulSet with 3 replicas. The thing is, I also want to use Consul Connect for service mesh — right now we have a homegrown mTLS setup that's fragile and nobody understands except me. Having a bus factor of one on the mTLS infrastructure is a risk."),
            ("assistant", "Consul Connect with transparent proxy would replace both your custom service discovery and the homegrown mTLS. Envoy sidecars handle the mTLS automatically, and service intentions give you an access control layer."),
            ("user", "Right, and the service intentions are declarative — I can define 'service A can talk to service B' in code rather than managing certificate rotations manually. Fundamentally this is about replacing bespoke infrastructure with community-supported tooling. We're not in the business of building service meshes."),
            ("assistant", "That's a strong argument for the review. The team's time should go toward business logic, not infrastructure that's already a solved problem. The migration path from your current setup to Consul would be the main thing to plan carefully."),
            ("user", "Agreed. I'll propose a phased migration — run Consul alongside the existing discovery for a month, migrate services one at a time, then decommission the old system. Same blue-green pattern we use for app deployments, just at the infrastructure level."),
        ],
    },
    {
        "topic": "side-project-cli-tool",
        "messages": [
            ("user", "I'm building a CLI tool in Go for managing my sourdough baking schedule. It tracks my starter feedings, calculates timing based on ambient temperature, and tells me when to do each step. I know it's over-engineered but it's fun."),
            ("assistant", "That's a great side project — it combines your engineering skills with your baking hobby. What's the data model look like?"),
            ("user", "The core model is a BakeSession with steps: feed starter, autolyse, mix, folds (with intervals), bulk ferment end, shape, proof, bake. Each step has a target time that's calculated from the start time and adjusted by temperature. The thing is, the timing isn't linear — fermentation rate roughly doubles with every 15°F increase, so I'm using an Arrhenius-style calculation."),
            ("assistant", "Using a temperature-dependent rate equation is clever. You could let the user input their kitchen temperature and the tool adjusts all the downstream timing. Does it account for the fridge retard as well?"),
            ("user", "Yeah, cold retard is modeled as a step with a different temperature — typically 38°F — so the fermentation rate drops dramatically. The tool calculates that a 4-hour room-temp bulk is roughly equivalent to 16 hours in the fridge. I store the bake sessions in a local SQLite database so I can track results over time and see which timing profiles produce the best loaves."),
            ("assistant", "Tracking results over time would let you build a feedback loop — adjust the model parameters based on actual outcomes. Very engineering-brained."),
            ("user", "Fundamentally it's the same feedback loop as any system — observe, measure, adjust. I'm using Cobra for the CLI framework and Charm for the terminal UI. The TUI shows a timeline of upcoming steps with countdown timers. My wife thinks I've lost it but the bread keeps getting better so she tolerates the madness."),
        ],
    },
    {
        "topic": "go-error-handling",
        "messages": [
            ("user", "I've been thinking about our error handling conventions in Go. We have a mix of sentinel errors, custom error types, and fmt.Errorf wrapping across different services. There's no consistency and it makes debugging harder than it should be."),
            ("assistant", "Inconsistent error handling is one of the most common Go codebase issues. A good starting point is defining a standard error package that all services import, with structured error types that include an error code, message, and cause chain."),
            ("user", "That's my plan. I want every error to carry a machine-readable code (like 'PAYMENT_DECLINED' or 'IDEMPOTENCY_CONFLICT') plus a human-readable message. The code is for programmatic handling, the message is for logs and debugging."),
            ("assistant", "That maps well to how gRPC status codes work — a numeric code plus a message. You could align your error codes with gRPC status codes where they overlap and add domain-specific codes for your business errors."),
            ("user", "Good idea. And I want errors to be wrappable — so 'PAYMENT_DECLINED' at the payment service becomes 'PAYMENT_DECLINED: upstream returned insufficient_funds' when it reaches the gateway. Each layer adds context without losing the original code."),
            ("assistant", "Go's error wrapping with %w supports this natively. Your custom error type would implement the Unwrap() method so errors.Is() and errors.As() work through the chain."),
            ("user", "Right. The thing is, I also need to make sure these errors serialize cleanly to JSON for our API responses and to our structured logging. One error type, three representations: Go error for internal handling, JSON for API responses, structured log fields for observability."),
        ],
    },
    {
        "topic": "fermentation-temperature-experiment",
        "messages": [
            ("user", "Running an experiment this weekend — baking three identical loaves at different bulk ferment temperatures: 72°F, 78°F, and 85°F. Same flour, same starter, same hydration percentage. Only variable is temperature."),
            ("assistant", "That's a great controlled experiment. The higher temperature should give you a faster ferment with more yeast activity (bigger bubbles, milder flavor), while the cooler one should be slower with more lactic acid production (tangier, tighter crumb). What are you measuring?"),
            ("user", "Bulk ferment duration to reach 50% volume increase, final crumb structure (photographed cross-section), crust color, and taste (blind test with my wife and the neighbors). I'm logging everything in a spreadsheet with timestamps every 30 minutes."),
            ("assistant", "You're applying A/B testing methodology to bread baking. The blind taste test is a nice touch — removes your own bias about which temperature you expected to produce the best result."),
            ("user", "Ha, exactly. The thing is, I already have a hypothesis — 78°F will produce the best balance of flavor and crumb. But the point of the experiment is to test the hypothesis, not confirm it. If 72°F wins I'll adjust my process."),
            ("assistant", "A true scientific approach. Are you controlling for shaping technique as well? That's another variable that heavily influences crumb structure."),
            ("user", "Same shaping for all three — my standard batard with a pre-shape rest. I'll shape them back to back so the technique is as consistent as possible. Fundamentally this is an exercise in isolating variables — same principle as debugging a distributed system. Change one thing, observe the result, iterate."),
        ],
    },
    {
        "topic": "home-network-redesign",
        "messages": [
            ("user", "Redesigning my home network. Currently everything is on one flat network — IoT devices, work laptop, kids' tablets, NAS, all on the same subnet. It's a security nightmare and the IoT devices are saturating the network with broadcast traffic."),
            ("assistant", "VLANs are the standard solution. You'd segment into at least three: a trusted network for your computers, an IoT VLAN for smart home devices, and a guest network. A managed switch and a router/firewall that supports VLANs (like pfSense or OPNsense) would handle this."),
            ("user", "I'm going with OPNsense on a small fanless PC — more control than a consumer router and I can run Suricata for IDS/IPS. The thing is, my Home Assistant server needs to be on both the trusted and IoT VLANs since it controls the smart home devices but I also access its dashboard from my laptop."),
            ("assistant", "You can give the Home Assistant server a second NIC or use 802.1Q VLAN tagging to put it on both VLANs. Alternatively, put it on the IoT VLAN and create a firewall rule allowing your trusted VLAN to access its web interface on port 8123."),
            ("user", "Firewall rule is cleaner — single NIC, single VLAN, explicit access. I'll also need to allow MQTT traffic from the IoT VLAN to reach the broker on the Home Assistant box. And mDNS relay for device discovery across VLANs."),
            ("assistant", "mDNS doesn't cross VLAN boundaries by default, so you'd need an mDNS reflector. OPNsense has an Avahi plugin that handles this — it rebroadcasts mDNS between specified VLANs."),
            ("user", "Good. Fundamentally this is the same zero-trust architecture we use at work — segment by trust level, allow only necessary traffic between segments, default deny everything else. My smart light bulbs don't need to talk to my NAS."),
        ],
    },
    {
        "topic": "api-rate-limiting",
        "messages": [
            ("user", "Need to add rate limiting to our public API. We've been running without it and last week a partner's integration bug sent us 50K requests per second for an hour. Our services handled it fine but the database connection pool was completely saturated."),
            ("assistant", "Rate limiting at the API gateway level would prevent this. You'd want per-API-key limits (e.g., 100 req/s per partner) plus a global limit as a safety net. Redis-based token bucket or sliding window algorithms are the standard approach."),
            ("user", "Token bucket is my preference — it allows bursts up to the bucket capacity while enforcing the average rate. I'll implement it in our Envoy gateway using the rate limit service pattern. The thing is, I also need different tiers — our premium partners get 500 req/s, standard gets 100, trial gets 20."),
            ("assistant", "You can map API key → tier in your auth layer and pass the tier as a header to the rate limit service. The rate limit service then applies the appropriate bucket configuration for that tier."),
            ("user", "Right. And I want to return proper 429 responses with Retry-After headers so well-behaved clients know when to back off. The partner that hammered us was actually retry-storming because their error handling treated 500s as transient and retried immediately with no backpressure."),
            ("assistant", "That's a classic retry storm. Rate limiting from your side protects you, but documenting exponential backoff with jitter in your API docs would help partners build more resilient integrations."),
            ("user", "Fundamentally this is a contract between us and our partners — we guarantee capacity up to the rate limit, they agree to respect 429s and back off. Without that contract it's just two systems fighting each other."),
        ],
    },
    {
        "topic": "cooking-pizza-dough",
        "messages": [
            ("user", "Made pizza dough this weekend using my sourdough discard. I usually throw the discard away but I figured the flavor profile would work for a longer-fermented pizza dough."),
            ("assistant", "Sourdough discard pizza dough is great — you get the tangy flavor without needing to plan around the starter's peak. How did you adjust the recipe for the discard?"),
            ("user", "I replaced about 20% of the flour and water with discard, added commercial yeast since the discard isn't at peak activity, and bumped the hydration percentage to 68% for a crispier crust. The thing is, the discard adds a slight sourness that plays really well with the tomato sauce."),
            ("assistant", "That's a nice balance — the commercial yeast handles the leavening reliability while the discard contributes flavor. 68% hydration should give you good extensibility for stretching without being too wet to handle."),
            ("user", "Exactly. I cold-fermented the dough balls for 48 hours which developed even more flavor. Baked on a preheated steel at 550°F — the crust was perfectly leopard-spotted with great spring. Fundamentally pizza dough is simpler than bread dough but the variables still matter — temperature, hydration, fermentation time."),
        ],
    },
    {
        "topic": "incident-postmortem",
        "messages": [
            ("user", "We had a P1 incident yesterday — payment processing was down for 23 minutes. Root cause was a config change that went out without the canary deployment catching it. I'm writing the postmortem now."),
            ("assistant", "23 minutes of payment downtime is significant. What was the config change and why didn't the canary catch it?"),
            ("user", "We changed the connection timeout for the payment provider client from 5 seconds to 500 milliseconds. The intent was to fail fast, but the provider has a legitimate cold start that takes ~800ms on the first request. The canary ran for 5 minutes but only saw warm requests since the provider was already hot. In production, pod restarts hit cold starts and every first request timed out."),
            ("assistant", "So the canary validated a steady-state scenario but missed the cold-start edge case. That's a subtle failure mode — the canary environment needs to replicate production conditions including restarts."),
            ("user", "Exactly. The thing is, we've been so focused on canary deployment for code changes that we forgot config changes need the same rigor. The fix was obvious — revert the timeout to 5 seconds — but the real fix is process: config changes go through canary with forced pod restarts during the canary window."),
            ("assistant", "That's a good corrective action. You might also want to add a synthetic probe that specifically tests the cold-start path — a health check that forces a fresh connection to the provider."),
            ("user", "Yeah, we're adding that. I'm also proposing that timeout values have minimum and maximum bounds enforced by validation — no one should be able to set a timeout below the provider's known cold-start latency. Fundamentally, the system should make it hard to introduce the same class of failure twice."),
            ("assistant", "Guardrails over guidelines. If the system enforces the constraint, you don't rely on humans remembering it."),
            ("user", "That's the principle. The postmortem action items are: 1) config canaries with forced restarts, 2) cold-start synthetic probes, 3) timeout bounds validation, 4) update the config change runbook. No blame, just better systems."),
        ],
    },
]

MEMORIES = [
    {"content": "Is a platform engineer at a fintech company working primarily in Go and Python. Builds event-driven microservices.", "updated_at": "2025-03-15"},
    {"content": "Has about 8 years of experience in distributed systems and backend engineering. Previously worked at a cloud infrastructure startup.", "updated_at": "2025-03-10"},
    {"content": "Prefers Go for performance-critical services and Python for data pipelines and scripting.", "updated_at": "2025-03-08"},
    {"content": "Uses Kubernetes, Argo CD, and Temporal in production. Strong opinions about service mesh design.", "updated_at": "2025-04-01"},
    {"content": "Currently migrating their team's architecture from synchronous REST to event-driven patterns using Kafka.", "updated_at": "2025-04-20"},
    {"content": "Is passionate about observability — uses Prometheus, Grafana, Loki, and is adding distributed tracing with OpenTelemetry and Tempo.", "updated_at": "2025-04-15"},
    {"content": "Strong believer in schema-first API design using Protocol Buffers and gRPC for internal services.", "updated_at": "2025-03-22"},
    {"content": "Read Designing Data-Intensive Applications by Kleppmann and considers it the best technical book they've encountered.", "updated_at": "2025-02-28"},
    {"content": "Is an avid sourdough baker. Maintains an active starter and bakes 2-3 loaves per week.", "updated_at": "2025-01-15"},
    {"content": "Built a proofing box using a seedling heat mat and Inkbird temperature controller for consistent fermentation.", "updated_at": "2025-02-01"},
    {"content": "Prefers a cold retard baking schedule that fits around their work schedule — mix in the evening, fridge overnight, bake in the morning.", "updated_at": "2025-03-05"},
    {"content": "Runs their home automation on Home Assistant with Zigbee2MQTT and ESPHome sensors, all communicating over MQTT.", "updated_at": "2025-02-20"},
    {"content": "Has approximately 30 ESPHome sensor nodes throughout their house monitoring temperature, humidity, motion, and door states.", "updated_at": "2025-03-12"},
    {"content": "Recently migrated smart home devices from WiFi to Zigbee for better reliability and local-only control.", "updated_at": "2025-04-05"},
    {"content": "Is building a whole-house energy monitoring system using SCT-013 CT clamps, ADS1115 ADCs, and ESPHome.", "updated_at": "2025-04-18"},
    {"content": "Signed up for a 50K trail race in the Cascades with 6,500 feet of elevation gain. Training with stair repeats and weekend trail runs.", "updated_at": "2025-04-25"},
    {"content": "Runs 30-40 miles per week, transitioning from road to trail training for the ultramarathon.", "updated_at": "2025-04-22"},
    {"content": "Uses systems thinking analogies for everything — baking, running, home automation. Frequently draws parallels between engineering and other domains.", "updated_at": "2025-03-30"},
    {"content": "Has a distinctive writing style: uses em-dashes, starts thoughts with 'the thing is' or 'fundamentally', and is concise but precise.", "updated_at": "2025-01-20"},
    {"content": "Is mentoring a junior developer on the team, focusing on teaching clean code and function extraction through pairing sessions.", "updated_at": "2025-04-10"},
    {"content": "Believes in two-tier on-call alerting: page-worthy (customer-impacting, immediate action) and ticket-worthy (next business day).", "updated_at": "2025-04-08"},
    {"content": "Had a P1 incident caused by a config change that bypassed canary deployment checks. Now advocates for config changes going through the same canary process as code changes.", "updated_at": "2025-04-28"},
    {"content": "Is evaluating Temporal as a replacement for their current workflow system built from cron jobs, SQS, and DynamoDB.", "updated_at": "2025-04-12"},
    {"content": "Building a CLI tool in Go (using Cobra and Charm) to manage sourdough baking schedules with temperature-dependent fermentation calculations.", "updated_at": "2025-03-25"},
    {"content": "Redesigned their home network with VLANs using OPNsense — separate trusted, IoT, and guest networks with explicit firewall rules between them.", "updated_at": "2025-02-10"},
    {"content": "Prefers to reason about systems in terms of feedback loops: observe, measure, adjust. Applies this to engineering, baking, and training.", "updated_at": "2025-02-15"},
    {"content": "Lives in the Pacific Northwest. Kitchen temperature drops to 66°F in fall, affecting starter and fermentation timing.", "updated_at": "2025-01-25"},
    {"content": "Uses PgBouncer for Postgres connection pooling in transaction mode to handle connection oversubscription during peak traffic.", "updated_at": "2025-03-18"},
    {"content": "Proposed replacing the team's custom service discovery with Consul and using Consul Connect for service mesh with mTLS.", "updated_at": "2025-04-02"},
    {"content": "Prefers sourdough at 75% hydration with bread flour. Working on improving crumb openness through longer autolyse and extended bulk fermentation.", "updated_at": "2025-02-25"},
    {"content": "Uses the principle 'change one variable at a time' in both debugging and baking experiments.", "updated_at": "2025-03-01"},
    {"content": "Ran a controlled experiment baking three loaves at different temperatures (72°F, 78°F, 85°F) with blind taste testing.", "updated_at": "2025-04-30"},
    {"content": "Makes pizza dough using sourdough discard at 68% hydration, cold-fermented for 48 hours, baked on a steel at 550°F.", "updated_at": "2025-04-27"},
    {"content": "Spouse is tolerant of the engineering approach to baking but occasionally points out the over-engineering.", "updated_at": "2025-02-05"},
    {"content": "Proposed implementing rate limiting using token bucket algorithm in Envoy with tiered limits per API key.", "updated_at": "2025-04-15"},
    {"content": "Experienced a retry storm from a partner integration that sent 50K requests per second. Uses this as a teaching example for backpressure design.", "updated_at": "2025-04-14"},
    {"content": "Values the principle 'guardrails over guidelines' — systems should enforce constraints rather than relying on humans to remember them.", "updated_at": "2025-04-29"},
    {"content": "Favorite technical concept: 'a log is the fundamental data structure — tables are just materialized views of a log.'", "updated_at": "2025-03-02"},
    {"content": "Plans to do a team book club reading DDIA one chapter per week.", "updated_at": "2025-03-03"},
    {"content": "Next technical book on the reading list is Database Internals by Petrov.", "updated_at": "2025-03-04"},
    {"content": "Thinks about ESPHome sensor nodes as microservices: single responsibility, well-defined MQTT interface, independently deployable via OTA.", "updated_at": "2025-03-13"},
    {"content": "Uses Grafana for both work dashboards and home energy monitoring visualizations.", "updated_at": "2025-04-19"},
    {"content": "Standard error handling pattern: machine-readable code, human-readable message, wrappable cause chain that serializes to JSON.", "updated_at": "2025-03-20"},
    {"content": "Running OPNsense with Suricata IDS/IPS on a fanless PC as their home firewall.", "updated_at": "2025-02-12"},
    {"content": "Has a Sonoff Zigbee coordinator running Zigbee2MQTT on a Raspberry Pi alongside Home Assistant.", "updated_at": "2025-02-22"},
    {"content": "Trains for the 50K by doing stair repeats (10x8 floors) at a parking garage near work.", "updated_at": "2025-04-23"},
    {"content": "Believes in the run-hike strategy for ultramarathons: optimize for total time, not per-mile pace.", "updated_at": "2025-04-24"},
    {"content": "Describes their engineering philosophy as: 'the system should make it hard to introduce the same class of failure twice.'", "updated_at": "2025-04-29"},
    {"content": "Uses Argo CD for GitOps-style deployments including monitoring alerting rules.", "updated_at": "2025-03-14"},
    {"content": "Working on standardizing Go error handling across all services with a shared error package.", "updated_at": "2025-03-21"},
    {"content": "Has three kids. Spouse and neighbors participate as blind taste testers for baking experiments.", "updated_at": "2025-01-10"},
    {"content": "Kitchen is equipped with a baking steel, digital scale, and proofing box. Takes cross-section photos of every loaf for crumb analysis.", "updated_at": "2025-02-02"},
    {"content": "Stores bake session data in a local SQLite database through their custom CLI tool to track results over time.", "updated_at": "2025-03-26"},
    {"content": "Describes circuit breakers as needing backpressure in the half-open state: exponential ramp-up, not instant close.", "updated_at": "2025-03-16"},
    {"content": "Compares Zigbee routers to Envoy sidecars in a service mesh — each mains-powered device routes messages for battery-powered end devices.", "updated_at": "2025-04-06"},
    {"content": "Uses the phrase 'same principle as' frequently when drawing analogies between different domains.", "updated_at": "2025-01-18"},
    {"content": "Prefers phased migrations: run new and old systems in parallel, migrate incrementally, then decommission. Calls it 'blue-green at the infrastructure level.'", "updated_at": "2025-04-03"},
    {"content": "Thinks about baking as debugging: 'the crumb doesn't lie' — outcome data tells the truth regardless of what you thought you did.", "updated_at": "2025-02-06"},
    {"content": "Is reading Neurofeedback literature for personal interest, particularly around focus and creativity.", "updated_at": "2025-01-28"},
    {"content": "Has a strong preference for local-first tools and privacy: runs Home Assistant locally, avoids cloud IoT, uses OPNsense instead of consumer routers.", "updated_at": "2025-02-13"},
    {"content": "Listens to technical podcasts during trail runs. Favorites include Software Engineering Daily and Oxide and Friends.", "updated_at": "2025-04-21"},
    {"content": "Uses SQLite for personal projects and ClickHouse for analytics workloads at work.", "updated_at": "2025-03-19"},
    {"content": "Applies the SRE book's framework for determining page-worthy alerts: 'what action do I need to take right now that can't wait 8 hours?'", "updated_at": "2025-04-09"},
    {"content": "Teaches junior developers using the Socratic method — asks questions rather than prescribing solutions.", "updated_at": "2025-04-11"},
    {"content": "Motto for code review: 'green tests first, then refactor' — get it working, then make it clean.", "updated_at": "2025-04-11"},
    {"content": "Uses BME280 sensors for temperature and humidity in all ESPHome nodes.", "updated_at": "2025-03-12"},
    {"content": "Has an automation that alerts if the garage door is open for more than 30 minutes.", "updated_at": "2025-03-13"},
    {"content": "Considers the sourdough autolyse analogous to a warm-up phase in distributed systems — preparation that improves the main operation.", "updated_at": "2025-02-26"},
    {"content": "Works remotely. Daily standup is at 8:30am which constrains baking schedules.", "updated_at": "2025-01-22"},
    {"content": "Has completed the Designing Data-Intensive Applications book club with the team and is considering Database Internals as the next one.", "updated_at": "2025-05-01"},
]


def generate_conversation_id():
    return str(uuid.uuid4())


def build_chatgpt_mapping(messages, conv_start):
    """Build a ChatGPT-style mapping tree from a flat message list.

    The ChatGPT export uses a tree where each node has an id, optional message,
    a parent pointer, and children list.  We build a simple linear chain:
      root (no message) -> system -> msg1 -> msg2 -> ...
    """
    mapping = {}
    node_ids = [str(uuid.uuid4()) for _ in range(len(messages) + 2)]

    # Root node (no message)
    root_id = node_ids[0]
    system_id = node_ids[1]
    mapping[root_id] = {
        "id": root_id,
        "message": None,
        "parent": None,
        "children": [system_id],
    }

    # System node
    mapping[system_id] = {
        "id": system_id,
        "message": {
            "id": system_id,
            "author": {"role": "system", "name": None, "metadata": {}},
            "create_time": None,
            "update_time": None,
            "content": {"content_type": "text", "parts": [""]},
            "status": "finished_successfully",
            "end_turn": True,
            "weight": 0,
            "metadata": {"is_visually_hidden_from_conversation": True},
            "recipient": "all",
            "channel": None,
        },
        "parent": root_id,
        "children": [node_ids[2]] if messages else [],
    }

    # Message nodes
    for j, (role, content) in enumerate(messages):
        nid = node_ids[j + 2]
        msg_time = conv_start + timedelta(minutes=j * 2 + 1)
        child_id = node_ids[j + 3] if j + 3 < len(node_ids) else None

        mapping[nid] = {
            "id": nid,
            "message": {
                "id": nid,
                "author": {"role": role, "name": None, "metadata": {}},
                "create_time": msg_time.timestamp(),
                "update_time": None,
                "content": {"content_type": "text", "parts": [content]},
                "status": "finished_successfully",
                "end_turn": role == "assistant",
                "weight": 1,
                "metadata": {
                    "request_id": uuid.uuid4().hex[:16] + "-SFO",
                    "message_source": None,
                    "timestamp_": "absolute",
                    "message_type": None,
                },
                "recipient": "all",
                "channel": None,
            },
            "parent": node_ids[j + 1],
            "children": [child_id] if child_id else [],
        }

    return mapping


def generate_conversations_json(out_dir):
    """Generate a single conversations.json in ChatGPT export format.

    This is the file users upload to the dashboard. parse_conversations.py
    converts it into per-conversation JSONL files.
    """
    base_time = datetime(2024, 9, 1, 10, 0, 0)
    export = []

    for i, conv in enumerate(CONVERSATIONS):
        conv_start = base_time + timedelta(days=i * 3, hours=i % 8)
        conv_end = conv_start + timedelta(minutes=len(conv["messages"]) * 2 + 5)

        mapping = build_chatgpt_mapping(conv["messages"], conv_start)

        export.append({
            "title": conv["topic"].replace("-", " ").title(),
            "create_time": conv_start.timestamp(),
            "update_time": conv_end.timestamp(),
            "mapping": mapping,
            "moderation_results": [],
            "current_node": list(mapping.keys())[-1],
            "conversation_id": str(uuid.uuid4()),
        })

    filepath = os.path.join(out_dir, "conversations.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print(f"  Generated conversations.json ({len(export)} conversations) in {filepath}")
    return len(export)


def generate_conversations(out_dir):
    """Generate per-conversation JSONL files (the parsed output format)."""
    conv_dir = os.path.join(out_dir, "conversations")
    os.makedirs(conv_dir, exist_ok=True)

    base_time = datetime(2024, 9, 1, 10, 0, 0)

    for i, conv in enumerate(CONVERSATIONS):
        conv_id = generate_conversation_id()
        filename = f"conversation_{conv_id}.jsonl"
        filepath = os.path.join(conv_dir, filename)

        conv_start = base_time + timedelta(days=i * 3, hours=i % 8)

        with open(filepath, "w", encoding="utf-8") as f:
            # System message
            sys_msg = {
                "timestamp": conv_start.isoformat(),
                "role": "system",
                "content": "",
                "tags": [],
                "line_number": 1,
            }
            f.write(json.dumps(sys_msg) + "\n")

            for j, (role, content) in enumerate(conv["messages"]):
                msg_time = conv_start + timedelta(minutes=j * 2 + 1)
                msg = {
                    "timestamp": msg_time.isoformat(),
                    "role": role,
                    "content": content,
                    "tags": [],
                    "line_number": j + 2,
                }
                f.write(json.dumps(msg) + "\n")

    print(f"  Generated {len(CONVERSATIONS)} JSONL files in {conv_dir}")
    return len(CONVERSATIONS)


def generate_memories(out_dir):
    mem_dir = os.path.join(out_dir, "memory")
    os.makedirs(mem_dir, exist_ok=True)

    memories_list = []
    for i, mem in enumerate(MEMORIES):
        memories_list.append({
            "id": uuid.uuid4().hex[:16],
            "content": mem["content"],
            "updated_at": mem["updated_at"],
        })

    output = {
        "memories": memories_list,
        "memory_max_tokens": 10000,
        "memory_num_tokens": sum(len(m["content"].split()) for m in MEMORIES) * 2,
    }

    filepath = os.path.join(mem_dir, "memories.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Generated {len(memories_list)} memories in {filepath}")
    return len(memories_list)


def count_user_messages():
    total = 0
    for conv in CONVERSATIONS:
        total += sum(1 for role, _ in conv["messages"] if role == "user")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for identity-mcp")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    print("Generating sample data for identity-mcp...")
    print(f"  Persona: Morgan Riley (platform engineer / sourdough baker)")
    print(f"  Output: {args.out}")
    print()

    n_export = generate_conversations_json(args.out)
    n_conv = generate_conversations(args.out)
    n_mem = generate_memories(args.out)
    n_user = count_user_messages()

    print()
    print(f"Done. {n_export} conversations in conversations.json, {n_conv} JSONL files,")
    print(f"      {n_user} user messages, {n_mem} memories.")
    print()
    print("To use this data:")
    print("  Option A (full pipeline):")
    print("    1. Copy sample_data/conversations.json -> conversations/conversations.json")
    print("    2. Copy sample_data/memory/memories.json -> memory/memories.json")
    print("    3. Run parse_conversations from the dashboard to generate JSONL files")
    print("    4. Run the rest of the pipeline (analyze_patterns, train_identity_model)")
    print()
    print("  Option B (skip parsing):")
    print("    1. Copy sample_data/conversations/*.jsonl -> conversations/")
    print("    2. Copy sample_data/memory/memories.json -> memory/memories.json")
    print("    3. Run train_identity_model directly")


if __name__ == "__main__":
    main()
