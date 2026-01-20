# X Algorithm Deep Dive: Complete Reference Guide

> Based on analysis of X's open-sourced algorithm (xAI's Grok-powered recommendation system)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Scoring Formula](#the-scoring-formula)
3. [Signal Weights](#signal-weights)
4. [The Two-Stage Pipeline](#the-two-stage-pipeline)
5. [Author Diversity Penalty](#author-diversity-penalty)
6. [Content Lifecycle](#content-lifecycle)
7. [Filtering System](#filtering-system)
8. [Strategic Recommendations](#strategic-recommendations)
9. [Technical Architecture](#technical-architecture)
10. [Quick Reference](#quick-reference)

---

## Executive Summary

The X algorithm is a sophisticated two-stage system that:

1. **Retrieves candidates** from in-network (Thunder) and out-of-network (Phoenix)
2. **Ranks everything** using a Grok-based transformer predicting 19 engagement types
3. **Applies weighted scoring** with author diversity and out-of-network penalties
4. **Filters aggressively** for quality, safety, and deduplication

### The Single Most Important Insight

**Likes are the primary ranking signal**, and negative feedback (blocks, mutes, reports, "not interested") actively suppresses your content. Create likeable, shareable content and avoid anything that triggers negative reactions.

---

## The Scoring Formula

The algorithm predicts **19 different engagement actions** and combines them into a single score:

```
Final Score = Σ (weight × P(action))
```

### Expanded Formula

```python
weighted_score = (
    + favorite_score × FAVORITE_WEIGHT
    + reply_score × REPLY_WEIGHT
    + repost_score × REPOST_WEIGHT
    + quote_score × QUOTE_WEIGHT
    + click_score × CLICK_WEIGHT
    + profile_click_score × PROFILE_CLICK_WEIGHT
    + vqv_score × VQV_WEIGHT  # video boost
    + share_score × SHARE_WEIGHT
    + share_via_dm_score × SHARE_VIA_DM_WEIGHT
    + share_via_copy_link_score × SHARE_VIA_COPY_LINK_WEIGHT
    + dwell_score × DWELL_WEIGHT
    + photo_expand_score × PHOTO_EXPAND_WEIGHT
    + follow_author_score × FOLLOW_AUTHOR_WEIGHT
    - not_interested_score × NOT_INTERESTED_WEIGHT
    - block_author_score × BLOCK_AUTHOR_WEIGHT
    - mute_author_score × MUTE_AUTHOR_WEIGHT
    - report_score × REPORT_WEIGHT
)
```

---

## Signal Weights

### Positive Signals (BOOST your content)

| Action | Index | Impact | Description |
|--------|-------|--------|-------------|
| **Likes (Favorites)** | 0 | **PRIMARY** | Highest weight - the main ranking metric |
| **Replies** | 1 | HIGH | Indicates conversation engagement |
| **Reposts** | 2 | HIGH | Viral/sharing signal |
| **Quotes** | 11 | HIGH | High-quality engagement |
| **Follow Author** | 13 | VERY HIGH | Ultimate quality signal |
| **Video Views (VQV)** | 6 | HIGH | Video gets special boost |
| **Profile Clicks** | 5 | MEDIUM | Author interest signal |
| **Shares (General)** | 7 | MEDIUM | Distribution signal |
| **DM Shares** | 8 | MEDIUM | Private sharing = quality signal |
| **Link Copy Shares** | 9 | MEDIUM | External sharing intent |
| **Dwell Time** | 10, 18 | MEDIUM | Time spent = quality signal |
| **Photo Expands** | 3 | LOW | Visual engagement |
| **Content Clicks** | 4 | LOW | Engagement depth |

### Negative Signals (SUPPRESS your content)

| Action | Index | Impact | Description |
|--------|-------|--------|-------------|
| **"Not Interested"** | 14 | -MEDIUM | Direct negative feedback |
| **Block Author** | 15 | -HIGH | **Severe penalty** |
| **Mute Author** | 16 | -HIGH | Strong negative signal |
| **Report** | 17 | -VERY HIGH | **Most damaging** action |

> **Critical:** One block/mute/report can devastate your reach across the platform.

---

## The Two-Stage Pipeline

### Stage 1: Retrieval (Candidate Sourcing)

**Thunder** (In-Network)
- Pulls recent posts from accounts you follow
- Content is ranked purely by timestamp (newest first)
- Posts have a natural advantage in your feed

**Phoenix Retrieval** (Out-of-Network)
- ML-based similarity search across ALL posts
- Uses embedding similarity to find relevant content
- Content receives OON_WEIGHT_FACTOR penalty (~0.7x)

### Stage 2: Ranking (Grok Transformer)

1. Takes your engagement history (128 recent actions)
2. Encodes each candidate with the transformer
3. Scores each candidate **independently** (candidate isolation mask)
4. Predicts probabilities for all 19 action types
5. Applies weighted combination
6. Applies diversity penalties

### The Scoring Pipeline (5 Sequential Scorers)

```
1. Phoenix Scorer    → ML predictions (19 probabilities)
2. Weighted Scorer   → Combine into single score
3. Author Diversity  → Penalize repeated authors
4. OON Scorer        → Penalize out-of-network content
5. Selection         → Top K posts by final score
```

---

## Author Diversity Penalty

If you post multiple times per day, your later posts receive exponentially decaying score multipliers.

### Decay Formula

```python
multiplier = max(FLOOR, BASE_MULTIPLIER ** (post_position - 1))
# Where BASE_MULTIPLIER ≈ 0.45 and FLOOR ≈ 0.10
```

### Score Decay by Post Position

| Post # | Multiplier | Effective Score |
|--------|------------|-----------------|
| 1st | 100% | Full score |
| 2nd | ~45% | Less than half |
| 3rd | ~32% | About a third |
| 4th | ~20% | One fifth |
| 5th | ~14% | Minimal |
| 6th+ | ~10% | Floor value |

**Takeaway:** Space out your posts. Quality > quantity.

---

## Content Lifecycle

### 48-Hour Window

Posts older than **48 hours** are automatically purged from the in-network cache. Your content has a 2-day lifespan to gain traction.

```
0h ────────── 24h ────────── 48h (purge)
│              │              │
Full reach    Declining     Removed from cache
```

### In-Network vs Out-of-Network

| Source | Multiplier | Description |
|--------|------------|-------------|
| In-Network | 1.0x | Posts from accounts you follow |
| Out-of-Network | ~0.7x | Posts from Phoenix ML retrieval |

Your followers see your content with a **natural advantage**.

---

## Filtering System

### Pre-Scoring Filters (Content becomes invisible)

1. **Duplicate posts** - Same content reposted
2. **Posts older than MAX_POST_AGE** - Stale content
3. **Self-posts** - You don't see your own posts
4. **Repost deduplication** - Only one version of reposted content
5. **Blocked/muted author** - You're invisible to those users
6. **Muted keywords** - Topic-based filtering
7. **Previously seen posts** - Uses Bloom filter + direct ID list
8. **Paywalled content** - If user isn't subscribed

### Post-Selection Filters

1. **Visibility filtering** - Spam, violence, gore, deleted content
2. **Conversation deduplication** - Only highest-scored post per thread

### Reply Filtering Rules

- Replies only appear if they're to posts from followed users
- Reply chains only show if the conversation root involves a followed user
- Random quote-tweet chains are filtered out

---

## Strategic Recommendations

### To Maximize Reach

| # | Strategy | Why It Works |
|---|----------|--------------|
| 1 | **Optimize for likes first** | Primary ranking signal |
| 2 | **Create shareable content** | DM shares and link copies signal quality |
| 3 | **Post native video** | Special VQV_WEIGHT boost |
| 4 | **Space your posts** | Avoid author diversity penalty |
| 5 | **Engage authentically** | Your history shapes recommendations to you |
| 6 | **Post within 48 hours of relevance** | Content expires from cache |
| 7 | **Build genuine followers** | In-network content has inherent advantage |
| 8 | **Encourage profile clicks** | Shows interest in you as an author |

### To Avoid Suppression

| # | Don't | Consequence |
|---|-------|-------------|
| 1 | Post block/mute/report-triggering content | Severe global penalty |
| 2 | Use clickbait or misleading headlines | "Not interested" signal |
| 3 | Spam posts | Author diversity penalty |
| 4 | Repost excessively | Deduplication filters |
| 5 | Ignore muted keyword topics | Invisible to those users |

---

## Technical Architecture

### Model Architecture (Grok Transformer)

- **Attention**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMS Normalization (instead of LayerNorm)
- **Activation**: GeLU in feed-forward layers
- **Query Attention**: Grouped (fewer KV heads than query heads)
- **Feed-Forward**: GLU-style networks
- **Logit Clipping**: Attention logits clipped at ±30

### Candidate Isolation

Each post is scored **independently** - it can't "see" other candidates.

Benefits:
- Scores are consistent and cacheable
- Your post's score doesn't depend on the batch
- Fair ranking regardless of competition

### Hash-Based Embeddings

User, post, and author embeddings use **multiple hash functions**:
- Provides robustness and collision resistance
- Hash value 0 is reserved for padding

### Engagement History

The Grok transformer encodes your last **128 engagement actions** to predict what you'll engage with next. Your recent behavior shapes your entire feed.

### Repository Structure

```
x-algorithm/
├── home-mixer/          # Orchestration layer (feed assembly)
├── thunder/             # In-network post store & retrieval
├── phoenix/             # ML scoring (Grok transformer)
└── candidate-pipeline/  # Reusable pipeline framework
```

---

## Quick Reference

### Signal Index Map

```
Index 0:  Favorite (Like)     Index 10: Dwell Time (Short)
Index 1:  Reply               Index 11: Quote
Index 2:  Repost              Index 12: (Reserved)
Index 3:  Photo Expand        Index 13: Follow Author
Index 4:  Content Click       Index 14: Not Interested
Index 5:  Profile Click       Index 15: Block Author
Index 6:  Video View (VQV)    Index 16: Mute Author
Index 7:  Share (General)     Index 17: Report
Index 8:  DM Share            Index 18: Dwell Time (Long)
Index 9:  Link Copy Share
```

### Key Numbers

| Metric | Value |
|--------|-------|
| Engagement history depth | 128 actions |
| Content cache TTL | 48 hours |
| Author diversity base | ~0.45 |
| Author diversity floor | ~0.10 |
| OON penalty | ~30% |
| Attention logit clip | ±30 |

### Priority Stack (What to Optimize)

1. **Likes** - PRIMARY metric
2. **Follow probability** - Ultimate quality signal
3. **Replies + Quotes** - High engagement signals
4. **Video views** - Special boost category
5. **Shares (all types)** - Distribution signals
6. **Avoid negatives** - Block/mute/report devastate reach

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial deep dive based on open-source algorithm analysis |

---

*This document is for educational purposes to help understand how content is ranked on X.*
