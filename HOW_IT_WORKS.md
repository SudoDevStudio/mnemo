# How Mnemo Works

Mnemo is a **middleman** that sits between your app and the AI service (like ChatGPT). It remembers answers to questions that have already been asked, so you don't pay for the same answer twice.

Think of it like a really smart assistant who takes notes. The first time someone asks a question, the assistant calls the expert (the AI). Every time after that, the assistant just reads from their notes — instantly, for free.

---

## The Basic Idea

Without Mnemo:

```
Your App  ──→  ChatGPT  ──→  Answer
               (costs money, takes 2-5 seconds)
```

With Mnemo:

```
Your App  ──→  Mnemo  ──→  Answer (from memory, instant, free)
                  │
                  └──→  ChatGPT (only if Mnemo hasn't seen this before)
```

**You change one line of code.** Instead of sending requests to `api.openai.com`, you send them to `localhost:8080`. Everything else stays exactly the same.

---

## A Real Example

Let's say you run a customer support chatbot for a shoe store.

### Monday morning

A customer asks: *"What is your return policy?"*

Mnemo has never seen this question. It forwards it to ChatGPT, gets the answer, and saves it. Cost: **$0.03**. Time: **2.1 seconds**.

### Monday afternoon

A different customer asks: *"What's your return policy?"*

Mnemo recognizes this is the same question. It returns the saved answer. Cost: **$0.00**. Time: **0.001 seconds**.

### Tuesday

A customer asks: *"How do I return something I bought?"*

This isn't word-for-word the same question, but it means the same thing. Mnemo is smart enough to recognize that. It returns the saved answer. Cost: **$0.00**. Time: **0.001 seconds**.

### The savings add up

If 500 customers ask about returns this month:
- **Without Mnemo:** 500 x $0.03 = **$15.00**, each waiting 2+ seconds
- **With Mnemo:** 1 x $0.03 = **$0.03**, 499 instant responses

---

## How It Recognizes "Same Question"

Mnemo has three layers of memory, like a person who checks their brain first, then their notebook, then their filing cabinet.

### Layer 1: Exact Match (the brain)

> "What is your return policy?" = "What is your return policy?"

Identical text? Instant match. This is the fastest check — takes 0.05 milliseconds (about 40,000x faster than blinking).

### Layer 2: Similar Meaning (the notebook)

> "What is your return policy?" ≈ "How do returns work?"

Not the same words, but the same meaning. Mnemo converts questions into numbers that represent their meaning (called embeddings) and compares them. If two questions are 92% similar in meaning, it's a match. Takes about 0.5 milliseconds.

### Layer 3: Long-Term Storage (the filing cabinet)

When you restart Mnemo, Layers 1 and 2 (which live in memory) are gone. Layer 3 saves important answers to a database (Redis) so they survive restarts. Takes about 2 milliseconds.

**The key thing:** Mnemo checks all three layers before ever calling the AI. Only on a complete miss — a question it's truly never seen before — does it call the AI.

---

## The Smart Parts

### It Learns Your Domain

A generic system treats "What is your return policy?" the same whether you're a shoe store, a hospital, or a bank. Mnemo adapts.

Over time, Mnemo notices patterns in your specific questions. A medical app's Mnemo gets better at recognizing when two medical questions mean the same thing. A coding app's Mnemo gets better at matching code-related questions.

This happens silently in the background. You don't configure it. It just gets more accurate the longer it runs.

### It Knows When Answers Go Stale

Imagine your return policy changes from "30 days" to "60 days". A dumb cache would keep serving the old answer forever.

Mnemo watches for signals that something is off:
- Are users correcting the answer or asking follow-ups?
- Has the AI started giving different answers to similar questions?
- Has it been a long time since this answer was verified?

When Mnemo suspects an answer is stale, it automatically throws it away and gets a fresh one next time.

### It Protects Expensive Answers

Not all AI responses cost the same. A simple greeting costs a fraction of a cent. A complex analysis with a large document might cost 50 cents.

When Mnemo's memory is full and it needs to make room, it asks: *"Which answer can I most afford to lose?"*

The answer it keeps longest is the one that:
- **Cost a lot to generate** (expensive to replace)
- **Gets asked frequently** (people need it)
- **Was recently used** (still relevant)
- **Isn't stale** (still correct)

A $0.50 answer that 100 people asked about today? Protected. A $0.001 greeting from last week? That goes first.

Here's what that looks like in practice:

| Cached Answer | Cost | Times Used | Age | Stale? | What Happens |
|---|---|---|---|---|---|
| Complex legal analysis | $0.50 | 100 times | 2 hours | No | **Kept** — expensive, popular, fresh |
| "Hello! How can I help?" | $0.001 | 5 times | 2 days | No | **Removed first** — cheap, rarely used |
| Outdated product specs | $0.10 | 50 times | 6 hours | Yes | **Removed soon** — stale info is dangerous |
| Fresh medical summary | $0.30 | 1 time | 5 min | No | **Kept** — expensive, even with low usage |

---

## What Happens When Something Goes Wrong

### The AI returns an error

Sometimes the AI service has problems — rate limits, outages, content filters. Mnemo **never saves error responses**. If the AI returns an error, Mnemo passes it through to your app but doesn't remember it. Next time someone asks the same question, Mnemo will try the AI again instead of replaying the error.

### Mnemo crashes or restarts

Layer 1 and 2 (in-memory) are lost. Layer 3 (Redis) survives. The most valuable answers are rebuilt from Layer 3 automatically. The cache warms up again as people ask questions.

### The AI service is completely down

Mnemo can still serve any answer it has in its cache. For popular questions, your app keeps working even during an AI outage. Only brand-new questions will fail.

---

## What Mnemo Is NOT

- **Not an AI itself.** Mnemo doesn't generate answers. It remembers them.
- **Not a router.** It doesn't pick which AI to use. It talks to whichever one you configure.
- **Not slow.** The overhead of checking the cache is under 2 milliseconds. You won't notice it.
- **Not risky.** If Mnemo is unsure whether a cached answer matches, it plays it safe and asks the AI. False matches are extremely rare (the similarity threshold is 92%).

---

## The Numbers

| Metric | Value |
|---|---|
| Overhead per request | Under 2ms |
| Layer 1 lookup speed | ~0.05ms |
| Layer 2 lookup speed | ~0.5ms |
| Layer 3 lookup speed | ~2ms |
| Default cache size | 10,000 exact + 100,000 semantic entries |
| Similarity threshold | 92% (configurable) |
| Works with | OpenAI, Anthropic, Vertex AI, Ollama, any OpenAI-compatible API |

---

## Running with Local AI (Ollama)

You don't need a cloud AI service at all. Mnemo works with [Ollama](https://ollama.com), which runs AI models directly on your computer — completely free, completely private.

```
Your App  ──→  Mnemo  ──→  Ollama (running on your machine)
```

Even though Ollama is free, caching still helps:
- **Speed** — A cached response takes 0.001 seconds. Even a local model takes 2-10 seconds to generate.
- **CPU/GPU relief** — Every cached response means your hardware isn't doing heavy inference work.
- **Consistency** — The same question always gets the same answer from cache (local models can vary between runs).

To set it up, just change your config to:

```yaml
upstream:
  provider: ollama
  base_url: http://localhost:11434
```

No API key needed. Everything else works the same.

---

## One Last Analogy

Imagine you hire a translator. Every time a customer speaks French, you call the translator ($$$, slow). After a while, you realize 80% of customers ask the same 20 questions.

So you hire a bilingual receptionist (Mnemo) who sits in front of the translator. The receptionist writes down every translation. When a customer asks something familiar — even if they phrase it slightly differently — the receptionist answers instantly from their notes.

The translator (AI) only gets called for genuinely new questions. Your phone bill drops. Your customers get faster answers. And the receptionist gets better at their job every day.

That's Mnemo.
