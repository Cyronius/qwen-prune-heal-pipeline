"""Training-only tool catalog.

Deliberately disjoint from bench/tools_catalog.json. None of these share a name or a
parameter schema with the seven tools the benchmark scores on, so a gain in tools_acc
after healing is evidence that tool-calling generalised rather than evidence that the
model memorised the test.

Some tools cover the same DOMAIN as an eval tool (weather, email, files). That is on
purpose. The model needs to relearn how to call tools, not to be starved of the subject
matter. What it must not see is the exact schema it will be scored on.

Structure: 18 domains, four tools each. Every domain contains at least one confusable
pair, because Round-1's damage was to tool SELECTION, not tool formatting. A catalog of
unambiguous tools would not exercise the broken behaviour.

Each tool carries request templates with {slot} placeholders. Slots are filled from the
pools in SLOTS, so the number of distinct prompts is combinatorial rather than equal to
the number of templates.
"""
from __future__ import annotations

import random

# Names in bench/tools_catalog.json. Nothing here may collide with these.
EVAL_TOOL_NAMES = {
    "get_weather", "get_forecast", "search_web", "create_calendar_event",
    "create_reminder", "send_email", "read_file",
}

SLOTS = {
    "city": ["Tokyo", "Lisbon", "Denver", "Oslo", "Lima", "Cairo", "Perth", "Hanoi",
             "Lagos", "Quebec City", "Bergen", "Chiang Mai", "Porto", "Reykjavik",
             "Wellington", "Tbilisi", "Marrakesh", "Ljubljana", "Busan", "Medellin"],
    "country": ["Japan", "Portugal", "Norway", "Peru", "Egypt", "Vietnam", "Georgia",
                "Iceland", "New Zealand", "Morocco", "Slovenia", "Colombia"],
    "person": ["Priya", "Marcus", "Yuki", "Ingrid", "Tomas", "Aisha", "Diego", "Nadia",
               "Ravi", "Lena", "Omar", "Freya", "Kwame", "Sofia", "Hiro", "Mira"],
    "email": ["priya@example.com", "m.becker@example.org", "yuki.tanaka@example.net",
              "ingrid@example.co", "tomas@example.io", "aisha.khan@example.com",
              "diego@example.org", "nadia@example.net", "ravi@example.io"],
    "date": ["2026-09-03", "2026-09-17", "2026-10-01", "2026-10-22", "2026-11-05",
             "2026-11-19", "2026-12-08", "2027-01-14", "2027-02-02", "2027-03-26"],
    "time": ["09:00", "10:30", "11:15", "13:00", "14:45", "16:00", "17:30", "19:00"],
    "days": ["3", "5", "7", "10", "2", "4"],
    "count": ["3", "5", "8", "10", "12", "20", "25", "50"],
    "hours": ["1.5", "2", "3.25", "0.75", "4", "6"],
    "file": ["notes.md", "config.yaml", "report.csv", "changelog.txt", "schema.sql",
             "budget.xlsx", "readme.rst", "pipeline.json", "deploy.sh"],
    "path": ["/srv/data/exports", "~/projects/atlas", "/var/log/ingest",
             "C:/work/reports", "./build/artifacts", "/opt/models/v3"],
    "repo": ["atlas-core", "ingest-pipeline", "billing-api", "web-console",
             "terraform-infra", "docs-site", "mobile-client"],
    "branch": ["main", "develop", "release/2.4", "fix/timeout-retry",
               "feat/bulk-upload", "chore/deps"],
    "amount": ["42.50", "1200", "89.99", "15000", "7.25", "340", "2499.00"],
    # separate pool where a big number would read as nonsense, e.g. a nightly hotel rate
    "small_amount": ["85", "120", "160", "210", "95", "140", "175"],
    "currency": ["USD", "EUR", "GBP", "JPY", "NOK", "BRL", "CAD"],
    "product": ["noise-cancelling headphones", "standing desk converter",
                "mechanical keyboard", "espresso grinder", "hiking boots",
                "portable monitor", "cast iron skillet", "merino base layer"],
    "artist": ["Bonobo", "Hiromi", "Sault", "Khruangbin", "Anouar Brahem",
               "Little Simz", "Nils Frahm", "Ibibio Sound Machine"],
    "song": ["Kerala", "Cage", "Wildfires", "Texas Sun", "Blue Maqams", "Introvert"],
    "room": ["living room", "kitchen", "bedroom", "office", "garage", "nursery",
             "basement", "hallway"],
    "device": ["front door camera", "hallway sensor", "patio speaker",
               "upstairs thermostat", "garage opener", "desk lamp"],
    "language": ["Portuguese", "Japanese", "Norwegian", "Arabic", "Vietnamese",
                 "Georgian", "Icelandic", "Slovene", "Swahili"],
    "metric": ["p99_latency", "error_rate", "queue_depth", "cpu_utilisation",
               "request_throughput", "cache_hit_ratio"],
    "service": ["checkout-api", "search-indexer", "auth-gateway", "media-transcoder",
                "notification-worker", "billing-sync"],
    "topic": ["retrieval-augmented generation", "post-quantum key exchange",
              "sourdough hydration ratios", "urban heat islands",
              "spaced repetition scheduling", "tidal energy turbines"],
    "company": ["Northwind Logistics", "Cobalt Labs", "Meridian Health",
                "Fairhaven Press", "Orchard Systems", "Tessellate AI"],
    "project": ["Atlas", "Beacon", "Cinder", "Driftwood", "Everest", "Foxglove"],
    "task_id": ["TASK-1042", "TASK-2318", "TASK-885", "TASK-4471", "TASK-609"],
    "order_id": ["ORD-88213", "ORD-40917", "ORD-15586", "ORD-72304"],
    "table": ["orders", "customers", "shipments", "invoices", "sessions", "audit_log"],
    "database": ["analytics", "prod_replica", "warehouse", "staging"],
    "url": ["https://example.org/whitepaper", "https://example.net/changelog",
            "https://example.com/pricing", "https://example.io/docs/api"],
    "airport": ["NRT", "LIS", "DEN", "OSL", "LIM", "CAI", "PER", "HAN", "YUL"],
    "dish": ["mushroom risotto", "lentil dahl", "grilled mackerel", "chickpea stew",
             "pad kee mao", "shakshuka"],
    "exercise": ["swimming", "trail running", "rowing", "cycling", "bouldering",
                 "kettlebell circuit"],
    "channel": ["#engineering", "#incidents", "#design-review", "#release-notes"],
    "flight": ["QR812", "SK4471", "LA2033", "NH106", "TP1049"],
    "query": ["failed payment retries", "orphaned session rows", "slow index scans",
              "duplicate shipment records"],
    "address": ["12 Harbour Lane, Bergen", "88 Rua Nova, Porto",
                "440 Alder Street, Denver", "7 Sakura Dori, Kyoto"],
}


# Slots where a repeated pair means "from X to Y" and so must come out in order.
ORDERED_SLOTS = {"date", "time"}


def fill(template, rng, used=None):
    """Substitute {slot} placeholders, without replacement within one template.

    A template like "from {branch} into {branch}" or "{amount} {currency} to
    {currency}" is nonsense if both draws land on the same value. Repeated slots
    therefore get distinct values, and date and time slots are sorted so a
    start-then-end pair reads in the right order.

    Pass a dict as `used` to record which values went where. The multi-turn builder
    needs that so a follow-up can change a value rather than restate it.
    """
    out = template
    for slot, pool in SLOTS.items():
        token = "{" + slot + "}"
        n = out.count(token)
        if not n:
            continue
        values = (rng.sample(pool, n) if n <= len(pool)
                  else [rng.choice(pool) for _ in range(n)])
        if slot in ORDERED_SLOTS:
            values.sort()
        if used is not None:
            used.setdefault(slot, []).extend(values)
        for value in values:
            out = out.replace(token, value, 1)
    return out


def _prop(spec, desc):
    if isinstance(spec, list):
        p = {"type": "string", "enum": spec}
    elif isinstance(spec, tuple):
        p = {"type": "array", "items": {"type": spec[1]}}
    else:
        p = {"type": spec}
    if desc:
        p["description"] = desc
    return p


class Tool:
    def __init__(self, name, domain, description, params, asks):
        if name in EVAL_TOOL_NAMES:
            raise ValueError(f"{name} collides with a benchmark tool")
        self.name, self.domain, self.description = name, domain, description
        self.params, self.asks = params, asks

    def schema(self):
        props = {k: _prop(spec, desc) for k, (spec, desc, _) in self.params.items()}
        required = [k for k, (_, _, req) in self.params.items() if req]
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": props, "required": required}}}

    def render_ask(self, rng):
        return fill(rng.choice(self.asks), rng)

    def slots_used(self):
        """Which {slot} names appear anywhere in this tool's request templates."""
        if not hasattr(self, "_slots"):
            self._slots = {s for s in SLOTS if any("{" + s + "}" in a for a in self.asks)}
        return self._slots


def T(name, domain, description, params, asks):
    return Tool(name, domain, description, params, asks)


S, I, N, B = "string", "integer", "number", "boolean"


def A(item="string"):
    return ("array", item)


TOOLS = [
    # ---- weather -------------------------------------------------------------
    T("get_current_conditions", "weather",
      "Retrieve present-moment weather readings for a location",
      {"location": (S, "City or place name", True),
       "unit_system": (["metric", "imperial"], "Measurement system", False)},
      ["What's it like outside in {city} right now?",
       "Give me current conditions for {city} in imperial units.",
       "Is it raining in {city} at the moment?"]),
    T("get_daily_forecast", "weather",
      "Retrieve a day-by-day weather outlook for the coming days",
      {"location": (S, "City or place name", True),
       "days_ahead": (I, "How many days to forecast, 1 to 14", True),
       "unit_system": (["metric", "imperial"], "Measurement system", False)},
      ["What's the weather going to do in {city} over the next {days} days?",
       "Give me a {days}-day outlook for {city}.",
       "Will it be dry in {city} this week? Look {days} days out."]),
    T("get_severe_alerts", "weather",
      "List active severe weather warnings issued for a region",
      {"region": (S, "Region, state or country", True),
       "min_severity": (["advisory", "watch", "warning"], "Lowest severity to include", False)},
      ["Any storm warnings active for {country}?",
       "Show me severe weather watches or worse for {country}.",
       "Are there weather alerts out for {country} today?"]),
    T("get_historical_weather", "weather",
      "Look up recorded weather observations for a past date range",
      {"location": (S, "City or place name", True),
       "start_date": (S, "ISO date YYYY-MM-DD", True),
       "end_date": (S, "ISO date YYYY-MM-DD", True)},
      ["What was the weather in {city} between {date} and {date}?",
       "Pull historical readings for {city} from {date} to {date}."]),

    # ---- calendar ------------------------------------------------------------
    T("schedule_meeting", "calendar",
      "Book a meeting on the shared calendar with one or more attendees",
      {"title": (S, "Meeting title", True),
       "start_date": (S, "ISO date YYYY-MM-DD", True),
       "start_time": (S, "24h time HH:MM", True),
       "duration_minutes": (I, "Length in minutes", False),
       "attendees": (A(), "Attendee email addresses", False)},
      ["Book a design review with {email} on {date} at {time}.",
       "Set up a 45 minute sync called 'Roadmap' on {date} at {time} with {email}.",
       "Put a meeting with {person} on the calendar for {date} at {time}."]),
    T("add_personal_reminder", "calendar",
      "Store a private reminder note for yourself with no attendees and no duration",
      {"note": (S, "Reminder text", True),
       "due_date": (S, "ISO date YYYY-MM-DD", True),
       "due_time": (S, "24h time HH:MM", False)},
      ["Remind me to renew my passport on {date}.",
       "Nudge me at {time} on {date} to call {person}.",
       "Set a personal note for {date}: order more coffee beans."]),
    T("find_free_slots", "calendar",
      "Find times when all listed participants are available on a given day",
      {"date": (S, "ISO date YYYY-MM-DD", True),
       "duration_minutes": (I, "Required slot length in minutes", True),
       "participants": (A(), "Participant email addresses", False)},
      ["When are {email} and I both free for 30 minutes on {date}?",
       "Find a 60 minute opening on {date} that works for {email}.",
       "What gaps do we have on {date} for an hour-long session?"]),
    T("cancel_event", "calendar",
      "Cancel a booked calendar event and optionally tell the attendees",
      {"event_title": (S, "Title of the event to cancel", True),
       "date": (S, "ISO date YYYY-MM-DD", True),
       "notify_attendees": (B, "Whether to email the attendees", False)},
      ["Cancel the standup on {date} and let everyone know.",
       "Drop my 'Roadmap' meeting on {date}, no need to notify anyone."]),

    # ---- messaging -----------------------------------------------------------
    T("compose_email", "messaging",
      "Draft and send an email message, optionally copying other recipients",
      {"recipient": (S, "Primary recipient address", True),
       "subject_line": (S, "Subject", True),
       "message_body": (S, "Body text", True),
       "cc": (A(), "Addresses to copy", False)},
      ["Write to {email} with the subject 'Invoice query' and ask when payment clears.",
       "Email {email} to say the {project} kickoff moved to {date}.",
       "Drop {email} a note titled 'Handover' explaining that {person} takes over Monday."]),
    T("send_chat_message", "messaging",
      "Post a message into a team chat channel or thread",
      {"channel": (S, "Channel name including the leading hash", True),
       "text": (S, "Message text", True),
       "thread_id": (S, "Thread to reply within", False)},
      ["Post in {channel} that the deploy is finished.",
       "Let {channel} know {service} is back to normal.",
       "Message {channel}: standup is cancelled tomorrow."]),
    T("search_mailbox", "messaging",
      "Search stored mail for messages matching a query",
      {"query": (S, "Search terms", True),
       "folder": (["inbox", "sent", "archive", "spam"], "Folder to search", False),
       "limit": (I, "Maximum messages to return", False)},
      ["Find emails mentioning {company} in my archive.",
       "Search my sent mail for anything about {topic}, top {count} only.",
       "Did {person} ever email me about the {project} budget?"]),
    T("set_out_of_office", "messaging",
      "Turn on an automatic away reply for a date range",
      {"start_date": (S, "ISO date YYYY-MM-DD", True),
       "end_date": (S, "ISO date YYYY-MM-DD", True),
       "message": (S, "Auto-reply text", True)},
      ["Set my away message from {date} to {date} saying I'm on leave.",
       "Turn on out of office between {date} and {date}, point people at {email}."]),

    # ---- files ---------------------------------------------------------------
    T("read_text_file", "files",
      "Load and return the text contents of a file",
      {"file_path": (S, "Full path to the file", True),
       "encoding": (S, "Character encoding, defaults to utf-8", False)},
      ["Show me what's in {path}/{file}.",
       "Open {path}/{file} and tell me what it says.",
       "What does {file} in {path} contain?"]),
    T("write_text_file", "files",
      "Write text content to a file, creating or replacing it",
      {"file_path": (S, "Full path to the file", True),
       "contents": (S, "Text to write", True),
       "overwrite": (B, "Replace the file if it already exists", False)},
      ["Save a note saying 'migration complete' to {path}/{file}, replacing it if needed.",
       "Write the deploy checklist into {path}/{file}."]),
    T("list_directory", "files",
      "List the entries inside a directory",
      {"directory_path": (S, "Directory to list", True),
       "recursive": (B, "Descend into subdirectories", False),
       "name_pattern": (S, "Glob pattern to filter names", False)},
      ["What's inside {path}?",
       "List every .csv under {path}, including subfolders.",
       "Show me the files in {path}."]),
    T("move_file", "files",
      "Move or rename a file from one path to another",
      {"source_path": (S, "Current path", True),
       "destination_path": (S, "New path", True),
       "overwrite": (B, "Replace the destination if it exists", False)},
      ["Move {path}/{file} into {path}.",
       "Rename {file} in {path} so it starts with archive-."]),

    # ---- search --------------------------------------------------------------
    T("web_search", "search",
      "Query a general internet search engine for current information",
      {"search_terms": (S, "What to search for", True),
       "max_results": (I, "Maximum results to return", False),
       "recency": (["any", "past_day", "past_week", "past_year"], "Freshness filter", False)},
      ["Look up recent coverage of {topic}.",
       "Search the web for {topic}, only results from the past week.",
       "Find me {count} sources about {topic}."]),
    T("search_internal_docs", "search",
      "Search the company's internal documentation and wiki spaces",
      {"query": (S, "Search terms", True),
       "space": (S, "Wiki space to restrict the search to", False),
       "limit": (I, "Maximum documents to return", False)},
      ["Search our internal wiki for the {service} runbook.",
       "Is there an internal doc about {topic}?",
       "Find the onboarding page for the {project} team."]),
    T("fetch_url_content", "search",
      "Download the content at a specific URL and return it",
      {"url": (S, "Full URL to fetch", True),
       "output_format": (["text", "html", "markdown"], "How to return the content", False)},
      ["Pull {url} down as markdown.",
       "Fetch the page at {url} and show me the text."]),
    T("find_images", "search",
      "Search for images matching a written description",
      {"description": (S, "What the image should show", True),
       "count": (I, "How many images to return", False),
       "license": (["any", "public_domain", "commercial_ok"], "License filter", False)},
      ["Find {count} commercially usable photos of {dish}.",
       "Get me public domain images of {city} at night."]),

    # ---- code ----------------------------------------------------------------
    T("open_pull_request", "code",
      "Open a pull request from one branch into another",
      {"repository": (S, "Repository name", True),
       "source_branch": (S, "Branch containing the changes", True),
       "target_branch": (S, "Branch to merge into", True),
       "title": (S, "Pull request title", True),
       "body": (S, "Description", False)},
      ["Open a PR in {repo} from {branch} into main called 'Retry on timeout'.",
       "Raise a pull request on {repo} merging {branch} to {branch}."]),
    T("list_open_issues", "code",
      "List issues that are currently open on a repository",
      {"repository": (S, "Repository name", True),
       "label": (S, "Filter to a single label", False),
       "assignee": (S, "Filter to one assignee", False)},
      ["What's still open on {repo}?",
       "Show me the bugs assigned to {person} in {repo}.",
       "List open issues tagged performance in {repo}."]),
    T("trigger_ci_run", "code",
      "Start a continuous integration workflow on a branch",
      {"repository": (S, "Repository name", True),
       "branch": (S, "Branch to build", True),
       "workflow": (S, "Workflow name", False)},
      ["Kick off CI for {branch} on {repo}.",
       "Re-run the nightly workflow on {repo} against main."]),
    T("get_commit_history", "code",
      "Retrieve recent commits on a branch",
      {"repository": (S, "Repository name", True),
       "branch": (S, "Branch to read", True),
       "limit": (I, "How many commits to return", False)},
      ["Show me the last {count} commits on {branch} in {repo}.",
       "What landed on main in {repo} recently?"]),

    # ---- database ------------------------------------------------------------
    T("run_sql_query", "database",
      "Execute a read-only SQL statement and return the result rows",
      {"sql": (S, "SQL statement to run", True),
       "database": (S, "Database name", True),
       "row_limit": (I, "Maximum rows to return", False)},
      ["Run a query on {database} to pull the {count} newest rows from {table}.",
       "Query {database} for {query}."]),
    T("describe_table", "database",
      "Return the column names and types of a table",
      {"database": (S, "Database name", True),
       "table": (S, "Table name", True)},
      ["What columns does {table} have in {database}?",
       "Describe the {table} table on {database}."]),
    T("export_table", "database",
      "Export an entire table to a file in the chosen format",
      {"database": (S, "Database name", True),
       "table": (S, "Table name", True),
       "output_format": (["csv", "parquet", "json"], "File format", True),
       "destination_path": (S, "Where to write the file", False)},
      ["Export {table} from {database} as parquet into {path}.",
       "Dump the {table} table to CSV."]),
    T("count_rows", "database",
      "Count rows in a table, optionally restricted by a condition",
      {"database": (S, "Database name", True),
       "table": (S, "Table name", True),
       "where_clause": (S, "SQL condition without the WHERE keyword", False)},
      ["How many rows are in {table} on {database}?",
       "Count the {table} records created after {date}."]),

    # ---- finance -------------------------------------------------------------
    T("send_payment", "finance",
      "Transfer money to a named recipient",
      {"recipient": (S, "Who to pay", True),
       "amount": (N, "Amount to send", True),
       "currency": (["USD", "EUR", "GBP", "JPY", "NOK", "CAD", "BRL"], "Currency code", True),
       "memo": (S, "Note attached to the payment", False)},
      ["Send {person} {amount} {currency} for the shared rental.",
       "Pay {company} {amount} {currency}, memo 'Q3 retainer'."]),
    T("get_account_balance", "finance",
      "Look up the current balance of an account",
      {"account_id": (S, "Account identifier", True),
       "currency": (S, "Currency to report in", False)},
      ["What's the balance on my main account?",
       "How much is left in the {project} budget account, in {currency}?"]),
    T("list_transactions", "finance",
      "List account transactions within a date range",
      {"account_id": (S, "Account identifier", True),
       "start_date": (S, "ISO date YYYY-MM-DD", True),
       "end_date": (S, "ISO date YYYY-MM-DD", True),
       "min_amount": (N, "Ignore transactions below this amount", False)},
      ["Show transactions on my main account between {date} and {date}.",
       "List anything over {amount} that hit the account since {date}."]),
    T("convert_currency", "finance",
      "Convert an amount from one currency to another at the current rate",
      {"amount": (N, "Amount to convert", True),
       "from_currency": (S, "Source currency code", True),
       "to_currency": (S, "Target currency code", True)},
      ["What's {amount} {currency} in {currency}?",
       "Convert {amount} {currency} to {currency} please."]),

    # ---- travel --------------------------------------------------------------
    T("search_flights", "travel",
      "Search available flights between two airports on a date",
      {"origin_airport": (S, "Departure airport code", True),
       "destination_airport": (S, "Arrival airport code", True),
       "departure_date": (S, "ISO date YYYY-MM-DD", True),
       "passengers": (I, "Number of travellers", False),
       "cabin": (["economy", "premium", "business"], "Cabin class", False)},
      ["Find flights from {airport} to {airport} on {date} for 2 people.",
       "Any business class seats {airport} to {airport} on {date}?"]),
    T("book_hotel", "travel",
      "Reserve a hotel room in a city for a date range",
      {"city": (S, "City to stay in", True),
       "check_in": (S, "ISO date YYYY-MM-DD", True),
       "check_out": (S, "ISO date YYYY-MM-DD", True),
       "guests": (I, "Number of guests", False),
       "max_nightly_price": (N, "Price ceiling per night", False)},
      ["Book somewhere in {city} from {date} to {date} for 2, under {small_amount} a night.",
       "Reserve a room in {city}, checking in {date} and out {date}."]),
    T("get_visa_requirements", "travel",
      "Check what entry documents a passport holder needs for a destination",
      {"passport_country": (S, "Country of the passport", True),
       "destination_country": (S, "Country being visited", True)},
      ["Do I need a visa for {country} on a {country} passport?",
       "What entry paperwork does a {country} citizen need for {country}?"]),
    T("track_flight_status", "travel",
      "Look up the live status of a specific flight",
      {"flight_number": (S, "Airline flight number", True),
       "date": (S, "ISO date YYYY-MM-DD", False)},
      ["Is {flight} on time?",
       "Check the status of flight {flight} on {date}."]),

    # ---- media ---------------------------------------------------------------
    T("play_track", "media",
      "Start playback of a specific song on a playback device",
      {"track": (S, "Song title", True),
       "artist": (S, "Performing artist", False),
       "device": (S, "Speaker or device to play on", False)},
      ["Play {song} by {artist} on the {room} speaker.",
       "Put {song} on."]),
    T("create_playlist", "media",
      "Create a new empty playlist",
      {"name": (S, "Playlist name", True),
       "description": (S, "Playlist description", False),
       "is_public": (B, "Whether others can see it", False)},
      ["Make me a private playlist called 'Deep Work'.",
       "Create a public playlist named 'Long Drives'."]),
    T("add_track_to_playlist", "media",
      "Append an existing song to a playlist that already exists",
      {"playlist_name": (S, "Target playlist", True),
       "track": (S, "Song title", True),
       "artist": (S, "Performing artist", False)},
      ["Add {song} by {artist} to my 'Deep Work' playlist.",
       "Stick {song} on the end of 'Long Drives'."]),
    T("get_recommendations", "media",
      "Suggest music similar to a seed artist",
      {"seed_artist": (S, "Artist to base suggestions on", True),
       "count": (I, "How many suggestions", False),
       "genre": (S, "Restrict to a genre", False)},
      ["Recommend {count} artists like {artist}.",
       "If I like {artist}, what else should I try?"]),

    # ---- smart home ----------------------------------------------------------
    T("set_thermostat", "home",
      "Set the target temperature for a room's thermostat",
      {"room": (S, "Which room", True),
       "target_temperature": (N, "Target temperature in degrees", True),
       "mode": (["heat", "cool", "auto"], "Operating mode", False)},
      ["Set the {room} to 20 degrees.",
       "Put the {room} thermostat on cool at 22."]),
    T("control_light", "home",
      "Switch a room's lights on or off and optionally set brightness",
      {"room": (S, "Which room", True),
       "state": (["on", "off"], "Desired state", True),
       "brightness_percent": (I, "Brightness from 0 to 100", False)},
      ["Turn the {room} lights off.",
       "Dim the {room} to 30 percent.",
       "Lights on in the {room} please."]),
    T("lock_door", "home",
      "Lock a specific door",
      {"door": (S, "Which door", True),
       "confirm": (B, "Require confirmation before locking", False)},
      ["Lock the front door.",
       "Make sure the garage door is locked."]),
    T("get_device_status", "home",
      "Report the current state of a smart home device",
      {"device": (S, "Device name", True),
       "room": (S, "Room the device is in", False)},
      ["Is the {device} online?",
       "What's the {device} doing right now?"]),

    # ---- maps ----------------------------------------------------------------
    T("get_directions", "maps",
      "Produce turn-by-turn directions between two places",
      {"origin": (S, "Starting point", True),
       "destination": (S, "Ending point", True),
       "mode": (["driving", "walking", "cycling", "transit"], "Travel mode", False),
       "avoid": (A(), "Things to avoid such as tolls or motorways", False)},
      ["How do I get from {address} to {address} by bike?",
       "Directions from {city} centre to the airport, avoiding tolls."]),
    T("find_nearby_places", "maps",
      "Find points of interest close to a location",
      {"near": (S, "Location to search around", True),
       "category": (S, "Type of place", True),
       "radius_metres": (I, "Search radius in metres", False),
       "open_now": (B, "Only include places currently open", False)},
      ["Find a pharmacy near {address} that's open now.",
       "Any bookshops within 500 metres of {address}?"]),
    T("estimate_travel_time", "maps",
      "Estimate how long a journey will take without returning the route itself",
      {"origin": (S, "Starting point", True),
       "destination": (S, "Ending point", True),
       "departure_time": (S, "24h time HH:MM", False),
       "mode": (["driving", "walking", "cycling", "transit"], "Travel mode", False)},
      ["How long does it take to get from {address} to the station if I leave at {time}?",
       "Roughly how far is {city} from the coast by car, in time?"]),
    T("geocode_address", "maps",
      "Convert a written address into latitude and longitude",
      {"address": (S, "Street address to resolve", True)},
      ["What are the coordinates of {address}?",
       "Geocode {address} for me."]),

    # ---- tasks ---------------------------------------------------------------
    T("create_task", "tasks",
      "Create a work item in a project tracker",
      {"title": (S, "Task title", True),
       "project": (S, "Project to file it under", True),
       "due_date": (S, "ISO date YYYY-MM-DD", False),
       "priority": (["low", "normal", "high", "urgent"], "Priority level", False),
       "assignee": (S, "Who it is assigned to", False)},
      ["File a high priority task on {project} to fix the {service} retry logic, due {date}.",
       "Add a task to {project}: write the migration runbook. Assign it to {person}."]),
    T("update_task_status", "tasks",
      "Change the status of an existing tracked task",
      {"task_id": (S, "Task identifier", True),
       "status": (["todo", "in_progress", "blocked", "done"], "New status", True),
       "comment": (S, "Note explaining the change", False)},
      ["Mark {task_id} as done.",
       "Move {task_id} to blocked and note that we're waiting on {company}."]),
    T("list_tasks", "tasks",
      "List tracked tasks matching a filter",
      {"project": (S, "Project to list from", True),
       "status": (["todo", "in_progress", "blocked", "done"], "Filter by status", False),
       "assignee": (S, "Filter by assignee", False),
       "limit": (I, "Maximum tasks to return", False)},
      ["What's in progress on {project}?",
       "Show me {person}'s open items on {project}.",
       "List the top {count} blocked tasks in {project}."]),
    T("log_time", "tasks",
      "Record hours worked against a task",
      {"task_id": (S, "Task identifier", True),
       "hours": (N, "Hours worked", True),
       "date": (S, "ISO date YYYY-MM-DD", False),
       "note": (S, "What the time was spent on", False)},
      ["Log {hours} hours to {task_id} for {date}.",
       "Put {hours} hours against {task_id}, note it was schema work."]),

    # ---- shopping ------------------------------------------------------------
    T("search_products", "shopping",
      "Search a retail catalogue for products",
      {"query": (S, "What to look for", True),
       "max_price": (N, "Price ceiling", False),
       "category": (S, "Product category", False),
       "in_stock_only": (B, "Exclude out of stock items", False)},
      ["Find {product} under {amount}, in stock only.",
       "Look for {product} in the catalogue."]),
    T("place_order", "shopping",
      "Order a product for delivery",
      {"product_id": (S, "Product identifier", True),
       "quantity": (I, "How many to order", True),
       "shipping_speed": (["standard", "express", "overnight"], "Delivery speed", False)},
      ["Order 2 of product SKU-4471 with express shipping.",
       "Buy 1 of SKU-8830, standard delivery is fine."]),
    T("track_shipment", "shopping",
      "Check where a placed order currently is",
      {"order_id": (S, "Order identifier", True)},
      ["Where's order {order_id}?",
       "Track {order_id} for me."]),
    T("start_return", "shopping",
      "Begin a return for a delivered order",
      {"order_id": (S, "Order identifier", True),
       "reason": (["damaged", "wrong_item", "no_longer_needed", "poor_quality"], "Return reason", True),
       "refund_method": (["original_payment", "store_credit"], "How to refund", False)},
      ["Return {order_id}, it arrived damaged.",
       "Start a return on {order_id} for store credit, wrong item shipped."]),

    # ---- monitoring ----------------------------------------------------------
    T("query_metric", "monitoring",
      "Fetch numeric time series values for a service metric",
      {"metric_name": (S, "Metric to query", True),
       "service": (S, "Service the metric belongs to", True),
       "start_time": (S, "ISO timestamp or relative offset", True),
       "end_time": (S, "ISO timestamp or relative offset", False),
       "aggregation": (["avg", "max", "min", "sum", "p95", "p99"], "How to aggregate", False)},
      ["What was p99 {metric} on {service} over the last hour?",
       "Chart {metric} for {service} since {date}."]),
    T("search_logs", "monitoring",
      "Search free-text log lines emitted by a service",
      {"service": (S, "Service to search", True),
       "query": (S, "Text to match in log lines", True),
       "min_severity": (["debug", "info", "warn", "error", "fatal"], "Lowest severity", False),
       "limit": (I, "Maximum lines to return", False)},
      ["Search {service} logs for {query}, errors and above.",
       "Any log lines in {service} mentioning timeout?"]),
    T("create_alert", "monitoring",
      "Create a threshold alert on a metric",
      {"metric_name": (S, "Metric to watch", True),
       "threshold": (N, "Value to compare against", True),
       "comparison": (["above", "below"], "Direction of the breach", True),
       "notify_channel": (S, "Where to send the alert", False)},
      ["Alert {channel} if {metric} goes above 500.",
       "Page us when {metric} drops below 0.9."]),
    T("get_service_health", "monitoring",
      "Report the overall health status of a service in an environment",
      {"service": (S, "Service name", True),
       "environment": (["dev", "staging", "production"], "Which environment", False)},
      ["Is {service} healthy in production?",
       "What's the status of {service} on staging?"]),

    # ---- contacts ------------------------------------------------------------
    T("find_contact", "contacts",
      "Look up a stored contact record",
      {"name": (S, "Person's name", False),
       "company": (S, "Company they work for", False),
       "email": (S, "Their email address", False)},
      ["Do we have contact details for {person} at {company}?",
       "Look up {email} in the contact list."]),
    T("create_contact", "contacts",
      "Save a new contact record",
      {"name": (S, "Person's name", True),
       "email": (S, "Email address", True),
       "company": (S, "Employer", False),
       "phone": (S, "Phone number", False)},
      ["Add {person} at {company} to contacts, email {email}.",
       "Save a new contact: {person}, {email}."]),
    T("log_interaction", "contacts",
      "Record that an interaction with a contact took place",
      {"contact_name": (S, "Who the interaction was with", True),
       "interaction_type": (["call", "email", "meeting", "demo"], "Kind of interaction", True),
       "notes": (S, "What happened", False),
       "date": (S, "ISO date YYYY-MM-DD", False)},
      ["Log a call with {person} on {date}, they want pricing.",
       "Record that I demoed to {person} at {company}."]),
    T("list_deals", "contacts",
      "List sales opportunities matching a filter",
      {"stage": (["prospect", "qualified", "proposal", "closed_won", "closed_lost"], "Pipeline stage", False),
       "owner": (S, "Deal owner", False),
       "min_value": (N, "Ignore deals below this value", False)},
      ["Show me deals in proposal stage worth more than {amount}.",
       "What's {person} got in the pipeline?"]),

    # ---- language ------------------------------------------------------------
    T("translate_text", "language",
      "Translate a passage of text into another language",
      {"text": (S, "Text to translate", True),
       "target_language": (S, "Language to translate into", True),
       "source_language": (S, "Language of the input", False),
       "formality": (["formal", "informal", "default"], "Register to use", False)},
      ["Translate 'the shipment has left the warehouse' into {language}, formally.",
       "Put this into {language}: we will follow up next week."]),
    T("detect_language", "language",
      "Identify which language a passage of text is written in",
      {"text": (S, "Text to analyse", True)},
      ["What language is 'kaj se dogaja danes'?",
       "Tell me which language this is: 'hvor er stasjonen'."]),
    T("transcribe_audio", "language",
      "Convert a recorded audio file into written text",
      {"file_path": (S, "Path to the audio file", True),
       "language": (S, "Spoken language", False),
       "include_timestamps": (B, "Add timestamps to the transcript", False)},
      ["Transcribe {path}/interview.wav with timestamps.",
       "Turn the recording at {path}/standup.m4a into text."]),
    T("summarize_document", "language",
      "Produce a shortened summary of a document already on disk",
      {"file_path": (S, "Path to the document", True),
       "max_words": (I, "Length ceiling for the summary", False),
       "style": (["bullets", "paragraph", "executive"], "Summary style", False)},
      ["Summarise {path}/{file} in bullets, under {count} words.",
       "Give me an executive summary of {file} in {path}."]),

    # ---- fitness -------------------------------------------------------------
    T("log_workout", "fitness",
      "Record a completed exercise session",
      {"activity": (S, "What exercise was done", True),
       "duration_minutes": (I, "How long it lasted", True),
       "intensity": (["easy", "moderate", "hard"], "Effort level", False),
       "date": (S, "ISO date YYYY-MM-DD", False)},
      ["Log 40 minutes of {exercise} at moderate effort.",
       "Record a hard {exercise} session on {date}, 55 minutes."]),
    T("get_activity_summary", "fitness",
      "Summarise recorded activity over a date range",
      {"start_date": (S, "ISO date YYYY-MM-DD", True),
       "end_date": (S, "ISO date YYYY-MM-DD", True),
       "metric": (["distance", "duration", "calories", "sessions"], "What to summarise", False)},
      ["How much did I train between {date} and {date}?",
       "Total distance from {date} to {date}, please."]),
    T("set_fitness_goal", "fitness",
      "Set a target to work towards by a deadline",
      {"metric": (["distance", "duration", "weight", "sessions"], "What is being targeted", True),
       "target_value": (N, "Target number", True),
       "deadline": (S, "ISO date YYYY-MM-DD", False)},
      ["Set a goal of 4 sessions a week until {date}.",
       "I want to hit 100 km of running by {date}."]),
    T("log_meal", "fitness",
      "Record something eaten",
      {"dish": (S, "What was eaten", True),
       "calories": (I, "Calorie count", False),
       "meal_type": (["breakfast", "lunch", "dinner", "snack"], "Which meal", False),
       "date": (S, "ISO date YYYY-MM-DD", False)},
      ["Log {dish} for lunch.",
       "Record that I had {dish} for dinner on {date}, about 600 calories."]),
]

BY_DOMAIN = {}
for _t in TOOLS:
    BY_DOMAIN.setdefault(_t.domain, []).append(_t)

# Cases where calling any tool is the wrong move. These mirror the nine shapes the
# benchmark's negative cases use, so the training data exercises the same judgement.
#
# Note that the benchmark deliberately shows search-style tools next to plain factual
# questions and still expects no call: the model is supposed to answer from what it
# knows. That is imitated here rather than avoided.
NEGATIVE_SLOTS = {
    "num_a": ["17", "38", "124", "9", "63", "251", "45", "78"],
    "num_b": ["23", "6", "12", "47", "19", "8", "31", "104"],
    "form": ["haiku", "limerick", "short poem", "two-line rhyme", "six-word story",
             "clerihew", "rhyming couplet", "very short fable", "one-line joke",
             "tiny riddle"],
    "subject": ["autumn leaves", "an empty station", "the last ferry", "a thunderstorm",
                "cold coffee", "a lighthouse in fog", "the first frost",
                "a cat on a warm roof", "a broken umbrella", "the night bus",
                "a library at closing time", "wet pavements", "a kettle boiling",
                "an unread letter", "the tide going out", "a queue in the rain",
                "a borrowed jumper", "the smell of rain on dust"],
    "concept": ["an HTTP 404 error", "a race condition", "a hash collision",
                "eventual consistency", "tail latency", "a memory leak",
                "idempotency", "a deadlock", "garbage collection",
                "a stack overflow", "a cache stampede", "back-pressure",
                "a circular dependency", "a null pointer dereference",
                "time complexity", "a merge conflict", "a foreign key constraint",
                "connection pooling", "rate limiting", "a heisenbug",
                "technical debt", "a feature flag", "load shedding",
                "a write-ahead log", "database sharding", "an off-by-one error",
                "dependency injection", "a semaphore", "blue-green deployment",
                "a regression test", "eventual vs strong consistency",
                "a bloom filter", "copy-on-write", "tail call optimisation"],
    "event": ["the Second World War end", "the Berlin Wall come down",
              "the euro enter circulation", "the first Moon landing happen",
              "the Bretton Woods system end", "the Suez Canal open",
              "the Chernobyl disaster occur", "the Titanic sink",
              "the first iPhone launch", "the Soviet Union dissolve",
              "the Channel Tunnel open", "the Apollo 13 mission fly",
              "the Great Fire of London happen", "the Panama Canal open",
              "the first heart transplant take place", "the Hubble telescope launch",
              "Concorde make its final flight", "the Magna Carta get signed",
              "the modern Olympic Games first take place",
              "the printing press reach Europe"],
    "trivia": ["the capital of Slovenia", "the tallest mountain in Africa",
               "the longest river in South America", "the smallest country in Europe",
               "the deepest ocean trench", "the largest desert on Earth",
               "the most spoken language in Brazil", "the currency used in Iceland",
               "the largest island in the Mediterranean",
               "the driest inhabited place on Earth",
               "the oldest continuously inhabited city",
               "the highest waterfall in the world", "the largest lake in Africa",
               "the northernmost capital city", "the only sea with no coastline",
               "the bird with the largest wingspan",
               "the hardest naturally occurring mineral",
               "the closest star to the Sun",
               "the coldest temperature ever recorded on Earth",
               "the longest bridge in Europe",
               "the number of bones in an adult human body",
               "the chemical symbol for tungsten",
               "the planet with the shortest day",
               "the largest volcano in the solar system",
               "the sea separating Europe from Africa"],
    "temp": ["100", "37", "212", "-40", "68", "451", "0", "98", "180", "425",
             "-15", "77", "350", "32"],
    "closer": ["Thanks, that's everything I needed!",
               "Great, that covers it. Thanks for the help.",
               "Perfect, nothing else for now.",
               "Cheers, I'm all set.",
               "That's all, appreciate it.",
               "Brilliant, no further questions.",
               "Lovely, we're done here.",
               "All good, thanks again.",
               "Understood, nothing more from me.",
               "That answers it, thank you.",
               "Right, I think that's us finished.",
               "No more questions, cheers."],
    "occasion": ["a launch party", "the quarterly offsite", "leaving drinks for Priya",
                 "a product demo day", "the summer picnic", "a retirement lunch",
                 "the release retrospective", "a welcome breakfast for new starters"],
    "sql_goal": ["find duplicate rows", "count orders per month",
                 "list customers with no orders", "find the slowest queries",
                 "spot gaps in an id sequence", "total revenue by region",
                 "find sessions that never ended", "rank products by return rate"],
    "capability": ["scheduling", "sending messages", "reading files", "looking things up",
                   "tracking orders", "monitoring services", "moving money",
                   "booking travel", "translating text", "logging workouts"],
    "draft_subject": ["a polite decline to a vendor",
                      "an apology for a late delivery",
                      "a request for a deadline extension",
                      "an introduction between two colleagues",
                      "a reminder that invoices are overdue",
                      "a thank-you note after an interview",
                      "an announcement that the office moves next month"],
    "everyday": ["why bread dough needs to rest before shaping",
                 "how a heat pump moves heat against a temperature gradient",
                 "the difference between arabica and robusta coffee",
                 "how noise-cancelling headphones work",
                 "why the sky is blue rather than violet",
                 "the difference between a moth and a butterfly",
                 "when to use 'fewer' rather than 'less'",
                 "why onions make you cry",
                 "why cast iron pans need seasoning",
                 "how a thermos keeps things hot and cold",
                 "why aeroplane windows have a tiny hole",
                 "how yeast makes bread rise",
                 "why the sea looks green in some places",
                 "what makes a knot hold rather than slip",
                 "why old photographs fade to orange",
                 "how sourdough starter stays alive",
                 "why some metals rust and others do not",
                 "what causes the smell after rain",
                 "why ice floats instead of sinking",
                 "how noise travels further at night",
                 "why cut apples turn brown",
                 "what makes bubbles in fizzy drinks stop",
                 "why the Moon looks bigger near the horizon",
                 "how compasses work near the poles",
                 "why cold water feels colder than cold air"],
    "inline": [
        "The new pipeline reduced build times from 40 minutes to 8 by caching dependencies.",
        "Attendance fell in the first quarter but recovered once the evening slot opened.",
        "The bridge closed for inspection after cracks appeared in two support columns.",
        "Sales of the mid-range model outpaced both the budget and premium versions.",
        "The pilot ran for six weeks and was extended after early results held up.",
        "Rainfall was well above average, which delayed planting across the region.",
        "Two suppliers merged, leaving a single source for the main component.",
        "Support tickets dropped once the onboarding guide was rewritten.",
        "The trial found no difference between the groups after twelve months.",
        "Membership grew steadily until the fee increase, then flattened.",
        "The old warehouse will be converted into flats starting next spring.",
        "Battery life improved by a third, though charging takes noticeably longer.",
        "Most respondents preferred the quieter model even at a higher price.",
        "The route was shortened by four miles but added two more stops.",
        "Costs came in under budget because the second phase was postponed.",
        "The festival moved indoors after three consecutive years of storms.",
    ],
}
SLOTS.update(NEGATIVE_SLOTS)

NEGATIVE_TEMPLATES = [
    # arithmetic the model should just do
    "What is {num_a} times {num_b}?",
    "What's {num_a} plus {num_b}, and is the result even?",
    "Is {num_a}{num_b} divisible by 3?",
    # creative writing
    "Write me a {form} about {subject}.",
    "Give me a {form} on the theme of {subject}.",
    # concept explanation
    "Explain what {concept} means.",
    "Describe {concept} to someone new to programming.",
    "In plain terms, what is {concept}?",
    # stable factual knowledge
    "What year did {event}?",
    "What is {trivia}?",
    "Tell me {trivia} without looking it up.",
    # everyday explanation
    "Explain {everyday}.",
    "I've always wondered about {everyday}. Can you explain?",
    # unit conversion done from knowledge
    "Convert {temp} degrees fahrenheit to celsius for me.",
    "How many kilometres is {num_a} miles?",
    # conversational closers
    "{closer}",
    # meta questions about the tools themselves
    "What tools do you have available? Just list them, don't use any.",
    "Which of these functions covers {capability}? Don't call anything, just say.",
    "Do any of your tools help with {capability}? Answer in words, no calls.",
    "Before you do anything: which function would you reach for if I asked about "
    "{capability}?",
    # explicit instruction not to act
    "Don't actually send anything, just draft me text inviting the team to {occasion}.",
    "Draft the wording for {draft_subject}. I'll do the sending myself later.",
    "Write me {draft_subject}, but don't send it anywhere.",
    "Don't run it. Just tell me what SQL you would write to {sql_goal}.",
    "Hold off on doing anything. What query would you use to {sql_goal}?",
    # summarising text supplied inline, which no file-based tool can take
    "Summarise this in one sentence: {inline}",
    "Rewrite this more formally: {inline}",
]


# Templates that read badly with an opener bolted on. A conversational closer is not
# something you prefix with "Quick one --".
NO_FRAME = {"{closer}", "What tools do you have available? Just list them, don't use any."}

NEGATIVE_OPENERS = ["", "", "Quick one -- ", "Before we start, ", "One more thing: ",
                    "If you don't mind, ", "Out of curiosity, ", "Just checking -- ",
                    "While I think of it, "]
NEGATIVE_CLOSERS = ["", "", " Keep it short.", " Thanks.", " A sentence or two is fine.",
                    " No need for detail.", " Don't overthink it.",
                    " Plain language please."]


def render_negative(rng):
    """A negative case, framed so the pool size is not the ceiling on distinct prompts.

    Single-slot templates cap out at the size of their pool. Wrapping them in an opener
    and a closer multiplies that by roughly 70, which is what lifts this slice from a
    few hundred distinct prompts to a few thousand.
    """
    template = rng.choice(NEGATIVE_TEMPLATES)
    text = fill(template, rng)
    if template in NO_FRAME:
        return text
    opener = rng.choice(NEGATIVE_OPENERS)
    closer = rng.choice(NEGATIVE_CLOSERS)
    if opener:
        text = opener + text[0].lower() + text[1:]
    return text + closer


# Kept for anything that wants a flat list rather than the generator.
NEGATIVE_ASKS = NEGATIVE_TEMPLATES

# Follow-up turns for the multi-turn slice, keyed by the slot they vary.
#
# The key matters. A follow-up has to change something the FIRST turn actually
# mentioned, or the exchange is incoherent -- "file a task on Foxglove" followed by
# "same thing but for Wellington" is not multi-turn data, it is noise, and the teacher
# would be recorded producing a confused answer to it.
#
# So a tool only gets a multi-turn case if one of its own request templates uses a slot
# that appears here. Tools with no match are skipped rather than given a bad follow-up.
FOLLOW_UPS_BY_SLOT = {
    "city": ["Same thing but for {city}.", "Now do {city}.", "And {city}?"],
    "country": ["Same question for {country}.", "What about {country}?"],
    "person": ["Do that again for {person}.", "Same for {person} please."],
    "email": ["Same again, but send it to {email}.", "Now do {email}."],
    "date": ["Actually, make it {date} instead.", "Change the date to {date}.",
             "Move it to {date}."],
    "time": ["Make it {time} instead.", "Actually, {time} works better."],
    "count": ["Change it to {count} this time.", "Make that {count} instead."],
    "project": ["And the same for {project}?", "Now do {project}."],
    "repo": ["Same thing on {repo}.", "Now for {repo}."],
    "branch": ["Do it for {branch} instead.", "Same but on {branch}."],
    "room": ["Now the {room} instead.", "Same for the {room}."],
    "device": ["Now check the {device}.", "Same for the {device}."],
    "service": ["Same for {service}.", "Now do {service}."],
    "table": ["Now do the {table} table.", "Same for {table}."],
    "database": ["Same query against {database}.", "Now on {database}."],
    "file": ["Same but for {file}.", "Now {file} instead."],
    "path": ["Same thing under {path}.", "Now look at {path}."],
    "currency": ["Show it in {currency} instead.", "Convert that to {currency}."],
    "amount": ["Make it {amount} instead.", "Change the amount to {amount}."],
    "song": ["Now play {song} instead.", "Same but {song}."],
    "artist": ["Do the same for {artist}.", "Now {artist}."],
    "address": ["Same for {address}.", "Now do {address}."],
    "task_id": ["Same for {task_id}.", "Now {task_id}."],
    "order_id": ["Now check {order_id}.", "Same for {order_id}."],
    "channel": ["Post it in {channel} too.", "Same message to {channel}."],
    "language": ["Now into {language}.", "Same but in {language}."],
    "exercise": ["Same but for {exercise}.", "Now log {exercise}."],
    "airport": ["Same but from {airport}.", "Now try {airport}."],
    "topic": ["Now look into {topic}.", "Same for {topic}."],
    "company": ["Same for {company}.", "Now {company}."],
}


def build_multiturn(tool, rng):
    """A two-turn exchange where the second turn changes something the first turn said.

    Three things have to line up or the pair is incoherent:
      - the follow-up must vary a slot that THIS request template used, not merely one
        the tool could have used in some other phrasing;
      - the new value must differ from the one already given, or the follow-up just
        restates the request;
      - the tool must have such a slot at all.

    Returns (first_turn, follow_up) or None when no coherent pair can be built.
    """
    used = {}
    template = rng.choice(tool.asks)
    first = fill(template, rng, used)
    candidates = [s for s in FOLLOW_UPS_BY_SLOT if "{" + s + "}" in template]
    rng.shuffle(candidates)
    for slot in candidates:
        fresh = [v for v in SLOTS[slot] if v not in used.get(slot, ())]
        if not fresh:
            continue
        shape = rng.choice(FOLLOW_UPS_BY_SLOT[slot])
        return first, shape.replace("{" + slot + "}", rng.choice(fresh))
    return None


def tools_with_follow_ups():
    return [t for t in TOOLS if t.slots_used() & FOLLOW_UPS_BY_SLOT.keys()]


# Back-compat: a flat list of every follow-up shape.
FOLLOW_UPS = [s for shapes in FOLLOW_UPS_BY_SLOT.values() for s in shapes]


def distinct_prompt_estimate():
    """Rough count of unique prompts the catalog can produce before repeating."""
    total = 0
    for tool in TOOLS:
        for ask in tool.asks:
            combos = 1
            for slot, pool in SLOTS.items():
                combos *= len(pool) ** ask.count("{" + slot + "}")
            total += combos
    return total


if __name__ == "__main__":
    rng = random.Random(0)
    print(f"{len(TOOLS)} tools across {len(BY_DOMAIN)} domains")
    print(f"{len(NEGATIVE_ASKS)} negative asks, {len(FOLLOW_UPS)} follow-up shapes")
    print(f"distinct positive prompts available: ~{distinct_prompt_estimate():,}")
    overlap = {t.name for t in TOOLS} & EVAL_TOOL_NAMES
    print(f"benchmark name overlap: {overlap or 'none'}")
    for domain, tools in BY_DOMAIN.items():
        print(f"\n[{domain}] {', '.join(t.name for t in tools)}")
        print(f"   e.g. {tools[0].render_ask(rng)}")
