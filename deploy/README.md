# 24/7 collector on AWS Lambda

Closes the hole GitHub Actions cannot: the instantaneous sources (MERIT,
Vidyut PRAVAH, UPSLDC, PSTCL) publish "now" and nothing else, and CI's
throttled ~12 runs a day gives 12 samples against the 96 we want. Every miss
is permanent.

Region is **ap-south-1 (Mumbai)** throughout — these are Indian government and
utility hosts, and an Indian IP removes any question about how they treat
foreign datacentre traffic.

## What it actually costs

Not zero. Close to it, and the honest number matters more than the round one.

| Service | Free allowance | This collector uses | Cost |
|---|---|---|---|
| Lambda requests | 1M/month, **permanent** | ~2,880 | 0 |
| Lambda compute | 400,000 GB-s/month, **permanent** | ~3,700 GB-s | 0 |
| EventBridge Scheduler | 14M invocations/month, **permanent** | ~2,880 | 0 |
| CloudWatch Logs | 5 GB ingest/month, permanent | ~1.4 MB | 0 |
| **S3 PUT** | 2,000/month, **first 12 months only** | ~11,500 | **~$0.06** |
| S3 GET | 20,000/month, first 12 months | ~17,300 | ~$0.01 |
| S3 storage | 5 GB, first 12 months | ~40 MB/month growing | <$0.01 |

**About $0.07/month — roughly Rs 6.** The Lambda and EventBridge halves are
free permanently; only S3 charges, and only because the collector makes more
small writes than the PUT allowance covers.

If you want a genuinely zero bill, DynamoDB's always-free tier (25 GB, 25 WCU)
covers this volume permanently. It costs a schema and an export step, which is
not worth Rs 6 a month before a demo. Revisit later if the volume grows.

## The trap that turns Rs 6 into Rs 2,800

**Do not attach the Lambda to a VPC.** A VPC-attached Lambda has no route to
the internet until you add a NAT Gateway, which is about $32/month billed by
the hour whether or not anything flows through it. Outside a VPC, Lambda gets
internet access for free. The console does not put you in a VPC unless you ask
for one, so the rule is simply: leave the VPC section alone.

Set an AWS Budget alert at $1 before anything else. If a bill ever appears,
that is the signal something is misconfigured, not that the estimate was wrong.

## Setup

### 1. Budget alarm first
Billing -> Budgets -> Create budget -> Zero spend budget -> your email.

### 2. S3 bucket
S3 -> Create bucket. Region **Asia Pacific (Mumbai) ap-south-1**. Name it
something unique, e.g. `flextrade-collected-<something>`. Every default is
fine — keep Block Public Access ON.

### 3. IAM role
IAM -> Roles -> Create role -> AWS service -> Lambda -> Next.
Attach `AWSLambdaBasicExecutionRole`. Name it `flextrade-collector-role`.

Then Add permissions -> Create inline policy -> JSON, replacing BUCKET:

```json
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject"],
  "Resource": "arn:aws:s3:::BUCKET/collected/*"}]}
```

Scoped to one prefix on one bucket. If the function is ever compromised it can
touch nothing else in the account.

### 4. Lambda function
Lambda -> Create function -> Author from scratch.
- Name `flextrade-collector`
- Runtime **Python 3.12**
- Architecture **arm64** (cheaper per GB-s, and we are inside the free tier either way)
- Permissions -> Use an existing role -> `flextrade-collector-role`
- **Leave the VPC section untouched**

Paste the whole of `deploy/lambda/handler.py` into the code editor, replacing
the stub. Click **Deploy**. No zip and no layer: the handler imports only the
standard library plus boto3, which the runtime already ships.

Configuration -> General configuration -> Edit:
- **Timeout 2 min** (MERIT is 23 sequential POSTs; the default 3s will fail)
- **Memory 256 MB**

Configuration -> Environment variables -> Add: `BUCKET` = your bucket name.

### 5. Test
Test tab -> Create a test event, any name, `{}` as the body -> Test.

Expect a log line like:

```json
{"at": "...", "ok": "5/5", "rows": 599,
 "detail": {"merit": 23, "npp_national": 539, "upsldc": 1, "pstcl": 1, "area_price": 35}}
```

Run it a second time. `npp_national` should drop to 0 — the rolling window is
already stored — while the instantaneous sources write again. If it does not,
dedupe is broken, not the network.

### 6. Schedule
EventBridge -> Scheduler -> Schedules -> Create schedule.
- Recurring, **Rate-based**, every **15 minutes**
- Flexible time window: **Off**
- Target: AWS Lambda Invoke -> `flextrade-collector`
- Permissions: let it create a new role
- **Turn OFF** "Delete schedule after completion"

### 7. Log retention
CloudWatch -> Log groups -> `/aws/lambda/flextrade-collector` -> Actions ->
Edit retention -> **30 days**. The default never expires and quietly
accumulates forever.

## Pulling the data back

Objects land at `s3://BUCKET/collected/<source>/<YYYY-MM-DD>.csv` — the same
columns the CI collector writes, so `ingest/merge_collected.py` reads them
unchanged once synced:

```
aws s3 sync s3://BUCKET/collected/ data/collected_aws/
```

## What still runs on GitHub Actions

Both collectors run. That is deliberate, not leftover.

The NPP endpoints serve a rolling ~4.1 hour window, so CI at roughly every two
hours already captures them with no loss; Lambda collecting them too means
neither runner being down costs a block. The instantaneous sources are the
ones only Lambda can keep at 15-minute cadence.

Two independent collectors writing the same schema is the cheapest redundancy
available, and this pipeline has already gone dark for eleven days once.
