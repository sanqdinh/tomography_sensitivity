---
name: hosting-and-domains
description: >-
  Committed hosting/ops decisions and how-tos for this project. Use WHENEVER the user
  asks about hosting, deployment, deploying, publishing/shipping the app, Fly.io, fly.toml,
  domains / domain names, DNS records, Porkbun, custom domains, SSL/TLS / HTTPS / certificates,
  scaling or VM sizing (memory/CPU), cold starts / scale-to-zero, hosting cost, regions, or
  CDN. Ground answers in the decisions below (deploy on Fly.io, domain via Porkbun); give
  Fly/Porkbun-specific concrete steps and do NOT re-pitch alternatives unless the user
  explicitly asks to reconsider.
---

# Hosting & domains — committed decisions for this project

The user has **decided**. When any hosting/domain/deployment/DNS/TLS/scaling/cost topic comes
up, treat these as settled and answer *within* them. Be concrete and Fly/Porkbun-specific.
Only revisit the platform choice if the user explicitly asks to reconsider (then see
"Alternatives considered" below for the rationale).

## The decisions

- **Hosting: Fly.io.** Scale-to-zero Docker container. Config lives in `fly.toml`
  (app `tomography-sensitivity`, region `ord`, `internal_port = 8501`).
- **Domain: `funtoimagine.org`, registered on Porkbun.** This project is served at the
  subdomain **`sendoe.funtoimagine.org`**, pointed at the Fly app via a **CNAME** →
  `tomography-sensitivity.fly.dev`, with Fly-managed Let's Encrypt TLS.
  - (Apex `funtoimagine.org` is not used by this project; the apex how-to below is kept only
    for reference in case the root is ever pointed here.)

### Why Fly (so you don't re-litigate it)
- **Docker-native** — runs the existing `Dockerfile` with the compiled IPOPT/k_aug binaries
  (`idaes get-extensions`) baked in. This requirement rules out pip-only platforms.
- **Scale-to-zero done right** — `min_machines_running = 0` + `concurrency.type = "connections"`
  keeps a machine alive while Streamlit's websocket streams a solve, so it is **not** autostopped
  mid-solve. Idle ⇒ ~$0.
- **No request timeout** — full VM, so multi-minute IPOPT solves just run (beats Cloud Run's
  60-min cap).
- **Private by default + tunable** — own/custom domain, pick memory/CPU, grow later.

## Fly: operating the deployment

```
fly deploy                       # build + ship (uses Dockerfile via fly.toml)
fly logs                         # tail; watch for OOM at higher image_res
fly status                       # machines / health
fly scale memory 4096            # bump RAM (see caveat below)
fly scale vm shared-cpu-4x       # faster CPU-bound solves
```

**Sizing caveat (from `CLAUDE.md`):** the app is CPU-bound (two IPOPT solves + k_aug) and cost
scales ~ `N² · n_horizon`. The default box is `shared-cpu-2x` / **2 GB**, which **OOMs when users
push `image_res` up**. If anyone but the owner will drive the dials, bump to `shared-cpu-4x` /
**4–8 GB**. 2 GB is fine for solo use at the default size.

**Cost model:** pay-as-you-go, **no free tier** (Fly removed free allowances in 2024). Scaled to
zero, a rarely-used prototype costs cents–a few $/mo. A dedicated IPv4 (often wanted for an apex
A record) is ~$2/mo; shared IPv4 is free.

## Custom domain: `sendoe.funtoimagine.org` (Porkbun → Fly)

This project uses a **subdomain**, so the path is a single CNAME — no IP allocation, no
dedicated IPv4. Flow (verified against Fly docs):

1. **Attach the subdomain to the app** (creates the cert request):
   ```
   fly certs add sendoe.funtoimagine.org
   fly certs setup sendoe.funtoimagine.org   # confirms the CNAME target to create
   ```

2. **Add ONE DNS record in Porkbun** (dashboard → `funtoimagine.org` → **Details / DNS**):
   - Type: **CNAME**
   - Host / Name: **`sendoe`**
   - Answer / Target: **`tomography-sensitivity.fly.dev`**
   - TTL: default
   - Porkbun adds default parking records on registration (an `ALIAS`/`*` wildcard → parking). The
     **explicit `sendoe` CNAME wins** over the `*` wildcard, so it just works; remove the wildcard
     only if it causes confusion.

3. **TLS is automatic** — once the CNAME resolves, Fly validates via TLS-ALPN and issues a
   **Let's Encrypt** cert. No manual cert work. `force_https = true` is already in `fly.toml`.

4. **Verify** (repeat until the cert shows issued; propagation is usually minutes):
   ```
   fly certs check sendoe.funtoimagine.org
   ```
   App is then live at **https://sendoe.funtoimagine.org**.

### Apex domains (reference only — not used by this project)
For a root domain you'd use **A + AAAA** records to the IPs from `fly ips list` (Porkbun also
supports **ALIAS** for CNAME-flattening at apex), and may want a dedicated IPv4
(`fly ips allocate-v4`, ~$2/mo; `--shared` for a free shared v4).

## Alternatives considered (rationale, if asked to reconsider)

- **Hugging Face Spaces (Docker)** — genuinely free, 16 GB RAM, but **public by default**,
  CPU-capped (2 vCPU), no SLA, and serves only on ports 80/443/**8080** (not 8501). Best for a
  free shareable demo; rejected here in favor of private + tunable.
- **Google Cloud Run** — real free tier + $300 trial and scale-to-zero, but **metered**, and its
  websocket billing means an open Streamlit tab keeps billing (not just during solves). Heavier
  setup. Viable fallback if enterprise/IAM is ever needed.
- Cold starts exist on **both** Fly (as configured, `min=0`) and HF free — not a differentiator.
  On Fly you can pay them away (`min_machines_running = 1`, or `suspend` instead of `stop`); on HF
  free you can't.
