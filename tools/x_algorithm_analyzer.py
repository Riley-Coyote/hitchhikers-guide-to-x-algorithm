#!/usr/bin/env python3
"""
X Algorithm Analyzer
====================

A Python tool for analyzing and predicting content engagement scores
based on X's open-sourced algorithm (xAI's Grok-powered recommendation system).

Features:
- Score calculation from engagement predictions
- Post content analysis for engagement estimation
- Author diversity penalty calculation
- Strategic recommendations
- Batch analysis for multiple posts

Usage:
    python x_algorithm_analyzer.py --help
    python x_algorithm_analyzer.py score --likes 0.5 --replies 0.2
    python x_algorithm_analyzer.py analyze "Your post text here"
    python x_algorithm_analyzer.py diversity --posts 5
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL WEIGHTS (Based on Algorithm Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

class SignalWeight:
    """Signal weights derived from X algorithm analysis."""

    # Positive signals (boost content)
    FAVORITE = 1.0        # PRIMARY ranking metric
    REPLY = 0.8           # High-value engagement
    REPOST = 0.75         # Viral/sharing signal
    QUOTE = 0.7           # High-quality engagement
    FOLLOW_AUTHOR = 1.2   # Ultimate quality signal
    VIDEO_VIEW = 0.9      # Video gets special boost
    PROFILE_CLICK = 0.5   # Author interest signal
    SHARE = 0.4           # Distribution signal
    DM_SHARE = 0.5        # Private sharing = quality
    LINK_COPY = 0.45      # External sharing intent
    DWELL_TIME = 0.3      # Time spent = quality
    PHOTO_EXPAND = 0.2    # Visual engagement
    CONTENT_CLICK = 0.15  # Engagement depth

    # Negative signals (suppress content)
    NOT_INTERESTED = -0.6  # Direct negative feedback
    BLOCK = -1.5           # SEVERE penalty
    MUTE = -1.2            # Strong negative signal
    REPORT = -2.0          # MOST damaging

    # Modifiers
    VIDEO_BONUS = 1.15     # Multiplier for native video
    OON_PENALTY = 0.7      # Out-of-network penalty

    # Author diversity
    DIVERSITY_BASE = 0.45
    DIVERSITY_FLOOR = 0.10


class SignalIndex(Enum):
    """Signal indices from the algorithm (0-18)."""
    FAVORITE = 0
    REPLY = 1
    REPOST = 2
    PHOTO_EXPAND = 3
    CONTENT_CLICK = 4
    PROFILE_CLICK = 5
    VIDEO_VIEW = 6
    SHARE = 7
    DM_SHARE = 8
    LINK_COPY = 9
    DWELL_SHORT = 10
    QUOTE = 11
    RESERVED = 12
    FOLLOW_AUTHOR = 13
    NOT_INTERESTED = 14
    BLOCK = 15
    MUTE = 16
    REPORT = 17
    DWELL_LONG = 18


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngagementProbabilities:
    """Predicted probabilities for each engagement action."""

    # Positive signals
    favorite: float = 0.0      # P(like)
    reply: float = 0.0         # P(reply)
    repost: float = 0.0        # P(repost)
    quote: float = 0.0         # P(quote)
    follow_author: float = 0.0 # P(follow)
    video_view: float = 0.0    # P(video view)
    profile_click: float = 0.0 # P(profile click)
    share: float = 0.0         # P(share)
    dm_share: float = 0.0      # P(DM share)
    link_copy: float = 0.0     # P(link copy)
    dwell_time: float = 0.0    # P(dwell)
    photo_expand: float = 0.0  # P(photo expand)
    content_click: float = 0.0 # P(content click)

    # Negative signals
    not_interested: float = 0.0  # P(not interested)
    block: float = 0.0           # P(block)
    mute: float = 0.0            # P(mute)
    report: float = 0.0          # P(report)

    def to_dict(self) -> Dict[str, float]:
        return {
            "favorite": self.favorite,
            "reply": self.reply,
            "repost": self.repost,
            "quote": self.quote,
            "follow_author": self.follow_author,
            "video_view": self.video_view,
            "profile_click": self.profile_click,
            "share": self.share,
            "dm_share": self.dm_share,
            "link_copy": self.link_copy,
            "dwell_time": self.dwell_time,
            "photo_expand": self.photo_expand,
            "content_click": self.content_click,
            "not_interested": self.not_interested,
            "block": self.block,
            "mute": self.mute,
            "report": self.report,
        }


@dataclass
class ContentModifiers:
    """Modifiers that affect the final score."""

    has_video: bool = False         # Native video content
    is_out_of_network: bool = False # OON content
    post_position: int = 1          # Position in author's daily posts (1-indexed)
    post_age_hours: float = 0.0     # Hours since posting
    has_image: bool = False         # Contains image
    has_link: bool = False          # Contains external link
    is_reply: bool = False          # Is a reply
    is_quote: bool = False          # Is a quote tweet


@dataclass
class ScoreResult:
    """Result of score calculation."""

    final_score: float
    positive_score: float
    negative_score: float
    diversity_multiplier: float
    oon_multiplier: float
    age_multiplier: float
    video_bonus_applied: bool
    interpretation: str
    recommendations: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class XAlgorithmScorer:
    """Calculate content scores based on X's algorithm."""

    def __init__(self):
        self.weights = SignalWeight()

    def calculate_score(
        self,
        probs: EngagementProbabilities,
        modifiers: ContentModifiers
    ) -> ScoreResult:
        """Calculate the final weighted score."""

        # Calculate positive contributions
        positive_score = (
            probs.favorite * self.weights.FAVORITE +
            probs.reply * self.weights.REPLY +
            probs.repost * self.weights.REPOST +
            probs.quote * self.weights.QUOTE +
            probs.follow_author * self.weights.FOLLOW_AUTHOR +
            probs.video_view * self.weights.VIDEO_VIEW +
            probs.profile_click * self.weights.PROFILE_CLICK +
            probs.share * self.weights.SHARE +
            probs.dm_share * self.weights.DM_SHARE +
            probs.link_copy * self.weights.LINK_COPY +
            probs.dwell_time * self.weights.DWELL_TIME +
            probs.photo_expand * self.weights.PHOTO_EXPAND +
            probs.content_click * self.weights.CONTENT_CLICK
        )

        # Calculate negative contributions (absolute value)
        negative_score = (
            probs.not_interested * abs(self.weights.NOT_INTERESTED) +
            probs.block * abs(self.weights.BLOCK) +
            probs.mute * abs(self.weights.MUTE) +
            probs.report * abs(self.weights.REPORT)
        )

        # Apply video bonus
        video_bonus_applied = False
        if modifiers.has_video and probs.video_view > 0:
            positive_score *= self.weights.VIDEO_BONUS
            video_bonus_applied = True

        # Calculate raw score
        raw_score = positive_score - negative_score

        # Calculate multipliers
        diversity_multiplier = self._calculate_diversity_multiplier(modifiers.post_position)
        oon_multiplier = self.weights.OON_PENALTY if modifiers.is_out_of_network else 1.0
        age_multiplier = self._calculate_age_multiplier(modifiers.post_age_hours)

        # Final score
        final_score = raw_score * diversity_multiplier * oon_multiplier * age_multiplier

        # Generate interpretation and recommendations
        interpretation = self._interpret_score(final_score, positive_score, negative_score)
        recommendations = self._generate_recommendations(probs, modifiers, final_score)

        return ScoreResult(
            final_score=round(final_score, 4),
            positive_score=round(positive_score, 4),
            negative_score=round(negative_score, 4),
            diversity_multiplier=round(diversity_multiplier, 4),
            oon_multiplier=round(oon_multiplier, 4),
            age_multiplier=round(age_multiplier, 4),
            video_bonus_applied=video_bonus_applied,
            interpretation=interpretation,
            recommendations=recommendations
        )

    def _calculate_diversity_multiplier(self, post_position: int) -> float:
        """Calculate author diversity penalty multiplier."""
        if post_position <= 1:
            return 1.0

        multiplier = self.weights.DIVERSITY_BASE ** (post_position - 1)
        return max(self.weights.DIVERSITY_FLOOR, multiplier)

    def _calculate_age_multiplier(self, age_hours: float) -> float:
        """Calculate age decay multiplier (48-hour window)."""
        if age_hours <= 0:
            return 1.0
        if age_hours >= 48:
            return 0.1  # Content effectively dead after 48 hours

        # Linear decay
        return max(0.1, 1 - (age_hours / 48))

    def _interpret_score(
        self,
        score: float,
        positive: float,
        negative: float
    ) -> str:
        """Generate human-readable interpretation."""

        if score < 0.3:
            return "LOW REACH: Content likely to be suppressed significantly"
        elif score < 0.6:
            return "MODERATE-LOW REACH: Content will struggle to gain traction"
        elif score < 1.0:
            return "MODERATE REACH: Content should reach some users"
        elif score < 1.5:
            return "GOOD REACH: Content is well-positioned in the algorithm"
        elif score < 2.0:
            return "EXCELLENT REACH: Content optimized for strong distribution"
        else:
            return "VIRAL POTENTIAL: Content has maximum algorithmic support"

    def _generate_recommendations(
        self,
        probs: EngagementProbabilities,
        modifiers: ContentModifiers,
        score: float
    ) -> List[str]:
        """Generate actionable recommendations."""

        recs = []

        # Check for high negative signals
        if probs.block > 0.05 or probs.mute > 0.05 or probs.report > 0.02:
            recs.append("WARNING: High negative signal probability. Review content for controversial elements.")

        if probs.not_interested > 0.1:
            recs.append("Consider making content more engaging or relevant to your audience.")

        # Author diversity
        if modifiers.post_position > 2:
            recs.append(f"You're on post #{modifiers.post_position} today. Consider spacing posts for better reach.")

        # OON penalty
        if modifiers.is_out_of_network:
            recs.append("Out-of-network content receives ~30% penalty. Build your follower base.")

        # Content age
        if modifiers.post_age_hours > 24:
            recs.append("Post is aging. Content has best reach within first 24 hours.")

        # Video opportunity
        if not modifiers.has_video and probs.video_view == 0:
            recs.append("Consider adding native video for additional algorithmic boost.")

        # Low engagement probability
        if probs.favorite < 0.3:
            recs.append("Low predicted likes. Focus on creating more likeable content (primary metric).")

        if probs.share < 0.05 and probs.dm_share < 0.02:
            recs.append("Low shareability. Create content worth sharing privately.")

        if score > 1.5 and len(recs) == 0:
            recs.append("Content is well-optimized! Continue this approach.")

        return recs


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class ContentAnalyzer:
    """Analyze post content to estimate engagement probabilities."""

    # Engagement indicators (heuristic-based)
    QUESTION_PATTERNS = [r'\?', r'what do you think', r'thoughts\?', r'agree\?']
    CALL_TO_ACTION = [r'retweet', r'rt if', r'share', r'like if', r'follow']
    CONTROVERSY_MARKERS = [r'hot take', r'unpopular opinion', r'controversial']
    POSITIVE_SENTIMENT = [r'amazing', r'incredible', r'love', r'best', r'great']
    NEGATIVE_TRIGGERS = [r'hate', r'worst', r'terrible', r'fight me', r'argue']

    def analyze_text(self, text: str) -> Tuple[EngagementProbabilities, ContentModifiers]:
        """Analyze text content and estimate engagement probabilities."""

        text_lower = text.lower()
        word_count = len(text.split())

        probs = EngagementProbabilities()
        modifiers = ContentModifiers()

        # Base probabilities (can be adjusted based on historical data)
        probs.favorite = 0.3
        probs.reply = 0.15
        probs.repost = 0.08
        probs.quote = 0.04
        probs.profile_click = 0.12
        probs.share = 0.05
        probs.dm_share = 0.02
        probs.dwell_time = 0.25

        # Check for questions (increases reply probability)
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower):
                probs.reply += 0.15
                break

        # Check for CTAs (increases share/repost probability)
        for pattern in self.CALL_TO_ACTION:
            if re.search(pattern, text_lower):
                probs.repost += 0.1
                probs.share += 0.05
                break

        # Check for positive sentiment (increases like probability)
        positive_count = sum(1 for p in self.POSITIVE_SENTIMENT if re.search(p, text_lower))
        probs.favorite += min(0.2, positive_count * 0.05)

        # Check for controversy markers (increases engagement but also negatives)
        for pattern in self.CONTROVERSY_MARKERS:
            if re.search(pattern, text_lower):
                probs.reply += 0.1
                probs.quote += 0.08
                probs.not_interested += 0.05
                probs.mute += 0.02
                break

        # Check for negative triggers
        negative_count = sum(1 for p in self.NEGATIVE_TRIGGERS if re.search(p, text_lower))
        if negative_count > 0:
            probs.block += min(0.05, negative_count * 0.02)
            probs.mute += min(0.08, negative_count * 0.03)
            probs.not_interested += min(0.15, negative_count * 0.05)

        # Check for media indicators
        if any(x in text_lower for x in ['http', 'pic.', 'video', '📹', '🎥']):
            modifiers.has_link = True

        if any(x in text_lower for x in ['📷', '🖼️', 'photo', 'image']):
            modifiers.has_image = True
            probs.photo_expand = 0.15

        if any(x in text_lower for x in ['video', '📹', '🎥', 'watch']):
            modifiers.has_video = True
            probs.video_view = 0.35

        # Adjust dwell time based on length
        if word_count > 50:
            probs.dwell_time += 0.15
        elif word_count < 10:
            probs.dwell_time -= 0.1

        # Check if reply/quote
        if text_lower.startswith('@'):
            modifiers.is_reply = True

        # Normalize probabilities to [0, 1]
        for attr in ['favorite', 'reply', 'repost', 'quote', 'follow_author',
                     'video_view', 'profile_click', 'share', 'dm_share', 'link_copy',
                     'dwell_time', 'photo_expand', 'content_click',
                     'not_interested', 'block', 'mute', 'report']:
            current = getattr(probs, attr)
            setattr(probs, attr, max(0.0, min(1.0, current)))

        return probs, modifiers


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class BatchAnalyzer:
    """Analyze multiple posts and calculate combined metrics."""

    def __init__(self):
        self.scorer = XAlgorithmScorer()
        self.content_analyzer = ContentAnalyzer()

    def analyze_posts(
        self,
        posts: List[str],
        is_same_author: bool = True
    ) -> Dict:
        """Analyze a batch of posts."""

        results = []

        for i, post in enumerate(posts):
            probs, modifiers = self.content_analyzer.analyze_text(post)

            if is_same_author:
                modifiers.post_position = i + 1

            score_result = self.scorer.calculate_score(probs, modifiers)

            results.append({
                "post_number": i + 1,
                "text_preview": post[:50] + "..." if len(post) > 50 else post,
                "score": score_result.final_score,
                "diversity_penalty": score_result.diversity_multiplier,
                "interpretation": score_result.interpretation,
            })

        # Summary statistics
        scores = [r["score"] for r in results]

        return {
            "post_count": len(posts),
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "best_score": round(max(scores), 4) if scores else 0,
            "worst_score": round(min(scores), 4) if scores else 0,
            "results": results,
            "recommendation": self._batch_recommendation(results, is_same_author)
        }

    def _batch_recommendation(self, results: List[Dict], is_same_author: bool) -> str:
        """Generate recommendation for batch."""

        if not results:
            return "No posts to analyze."

        if is_same_author and len(results) > 3:
            return (f"Warning: {len(results)} posts from same author. "
                    "Posts 4+ receive <20% of normal score. "
                    "Consider spacing posts throughout the day.")

        avg_score = sum(r["score"] for r in results) / len(results)

        if avg_score < 0.5:
            return "Overall low engagement predicted. Review content strategy."
        elif avg_score < 1.0:
            return "Moderate engagement expected. Focus on increasing likeability."
        else:
            return "Good engagement potential across posts."


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    """Print tool header."""
    print("\n" + "═" * 60)
    print("  X ALGORITHM ANALYZER")
    print("  Score Calculator & Strategy Tool")
    print("═" * 60 + "\n")


def print_score_result(result: ScoreResult):
    """Pretty print score result."""

    print("┌" + "─" * 40 + "┐")
    print(f"│ FINAL SCORE: {result.final_score:>24.4f} │")
    print("├" + "─" * 40 + "┤")
    print(f"│ Positive Score:     {result.positive_score:>17.4f} │")
    print(f"│ Negative Score:     {result.negative_score:>17.4f} │")
    print(f"│ Diversity Mult:     {result.diversity_multiplier:>17.4f} │")
    print(f"│ OON Multiplier:     {result.oon_multiplier:>17.4f} │")
    print(f"│ Age Multiplier:     {result.age_multiplier:>17.4f} │")
    print(f"│ Video Bonus:        {'Yes' if result.video_bonus_applied else 'No':>17} │")
    print("└" + "─" * 40 + "┘")

    print(f"\n📊 {result.interpretation}\n")

    if result.recommendations:
        print("💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")


def cmd_score(args):
    """Handle score command."""

    probs = EngagementProbabilities(
        favorite=args.likes,
        reply=args.replies,
        repost=args.reposts,
        quote=args.quotes,
        follow_author=args.follow,
        video_view=args.video_views,
        profile_click=args.profile_clicks,
        share=args.shares,
        dm_share=args.dm_shares,
        dwell_time=args.dwell,
        not_interested=args.not_interested,
        block=args.block,
        mute=args.mute,
        report=args.report
    )

    modifiers = ContentModifiers(
        has_video=args.has_video,
        is_out_of_network=args.oon,
        post_position=args.post_position,
        post_age_hours=args.age
    )

    scorer = XAlgorithmScorer()
    result = scorer.calculate_score(probs, modifiers)

    print_header()
    print_score_result(result)

    if args.json:
        print("\n📋 JSON Output:")
        print(json.dumps({
            "score": result.final_score,
            "positive": result.positive_score,
            "negative": result.negative_score,
            "multipliers": {
                "diversity": result.diversity_multiplier,
                "oon": result.oon_multiplier,
                "age": result.age_multiplier
            },
            "interpretation": result.interpretation,
            "recommendations": result.recommendations
        }, indent=2))


def cmd_analyze(args):
    """Handle analyze command."""

    analyzer = ContentAnalyzer()
    scorer = XAlgorithmScorer()

    probs, modifiers = analyzer.analyze_text(args.text)

    # Apply any CLI overrides
    modifiers.post_position = args.post_position
    modifiers.post_age_hours = args.age
    modifiers.is_out_of_network = args.oon

    result = scorer.calculate_score(probs, modifiers)

    print_header()
    print(f"📝 Analyzing: \"{args.text[:80]}{'...' if len(args.text) > 80 else ''}\"\n")

    print("Estimated Engagement Probabilities:")
    print(f"  Likes: {probs.favorite:.2%}  |  Replies: {probs.reply:.2%}  |  Reposts: {probs.repost:.2%}")
    print(f"  Shares: {probs.share:.2%}  |  Profile Clicks: {probs.profile_click:.2%}")
    print(f"  Video Views: {probs.video_view:.2%}  |  Dwell: {probs.dwell_time:.2%}")

    if probs.block > 0 or probs.mute > 0 or probs.report > 0:
        print(f"\n⚠️  Negative Signals Detected:")
        print(f"  Block: {probs.block:.2%}  |  Mute: {probs.mute:.2%}  |  Report: {probs.report:.2%}")

    print()
    print_score_result(result)


def cmd_diversity(args):
    """Handle diversity command."""

    print_header()
    print("📉 Author Diversity Penalty Analysis\n")

    base = SignalWeight.DIVERSITY_BASE
    floor = SignalWeight.DIVERSITY_FLOOR

    print("Post Position  |  Multiplier  |  Effective Score")
    print("─" * 50)

    for i in range(1, args.posts + 1):
        mult = max(floor, base ** (i - 1))
        effective = mult * 100
        bar = "█" * int(effective / 5) + "░" * (20 - int(effective / 5))
        print(f"    Post {i:2d}     |    {mult:.2%}    |  {bar} {effective:.1f}%")

    print("\n💡 Key insight: Space your posts. Quality > quantity.")
    print(f"   Posts after #{int(2-1 + (1/base))} receive diminishing returns.")


def cmd_batch(args):
    """Handle batch command."""

    if args.file:
        with open(args.file, 'r') as f:
            posts = [line.strip() for line in f if line.strip()]
    else:
        posts = args.posts

    analyzer = BatchAnalyzer()
    results = analyzer.analyze_posts(posts, is_same_author=args.same_author)

    print_header()
    print(f"📊 Batch Analysis ({results['post_count']} posts)\n")

    print(f"Average Score: {results['average_score']}")
    print(f"Best Score: {results['best_score']}")
    print(f"Worst Score: {results['worst_score']}")

    print("\n" + "─" * 60)
    for r in results['results']:
        print(f"Post {r['post_number']}: {r['score']:.4f} (×{r['diversity_penalty']:.2f}) - {r['text_preview']}")
    print("─" * 60)

    print(f"\n💡 {results['recommendation']}")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="X Algorithm Analyzer - Score Calculator & Strategy Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python x_algorithm_analyzer.py score --likes 0.5 --replies 0.2
  python x_algorithm_analyzer.py analyze "Check out my new video!"
  python x_algorithm_analyzer.py diversity --posts 10
  python x_algorithm_analyzer.py batch --file posts.txt
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Score command
    score_parser = subparsers.add_parser('score', help='Calculate score from engagement probabilities')
    score_parser.add_argument('--likes', type=float, default=0.3, help='P(like)')
    score_parser.add_argument('--replies', type=float, default=0.15, help='P(reply)')
    score_parser.add_argument('--reposts', type=float, default=0.08, help='P(repost)')
    score_parser.add_argument('--quotes', type=float, default=0.04, help='P(quote)')
    score_parser.add_argument('--follow', type=float, default=0.02, help='P(follow)')
    score_parser.add_argument('--video-views', type=float, default=0.0, help='P(video view)')
    score_parser.add_argument('--profile-clicks', type=float, default=0.12, help='P(profile click)')
    score_parser.add_argument('--shares', type=float, default=0.05, help='P(share)')
    score_parser.add_argument('--dm-shares', type=float, default=0.02, help='P(DM share)')
    score_parser.add_argument('--dwell', type=float, default=0.25, help='P(dwell time)')
    score_parser.add_argument('--not-interested', type=float, default=0.02, help='P(not interested)')
    score_parser.add_argument('--block', type=float, default=0.01, help='P(block)')
    score_parser.add_argument('--mute', type=float, default=0.01, help='P(mute)')
    score_parser.add_argument('--report', type=float, default=0.0, help='P(report)')
    score_parser.add_argument('--has-video', action='store_true', help='Content has native video')
    score_parser.add_argument('--oon', action='store_true', help='Out-of-network content')
    score_parser.add_argument('--post-position', type=int, default=1, help='Post position today')
    score_parser.add_argument('--age', type=float, default=0, help='Post age in hours')
    score_parser.add_argument('--json', action='store_true', help='Output JSON')
    score_parser.set_defaults(func=cmd_score)

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze post text')
    analyze_parser.add_argument('text', help='Post text to analyze')
    analyze_parser.add_argument('--post-position', type=int, default=1, help='Post position today')
    analyze_parser.add_argument('--age', type=float, default=0, help='Post age in hours')
    analyze_parser.add_argument('--oon', action='store_true', help='Out-of-network content')
    analyze_parser.set_defaults(func=cmd_analyze)

    # Diversity command
    diversity_parser = subparsers.add_parser('diversity', help='Show author diversity penalty')
    diversity_parser.add_argument('--posts', type=int, default=10, help='Number of posts to show')
    diversity_parser.set_defaults(func=cmd_diversity)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple posts')
    batch_parser.add_argument('--file', help='File with posts (one per line)')
    batch_parser.add_argument('--posts', nargs='+', help='Posts to analyze')
    batch_parser.add_argument('--same-author', action='store_true', default=True,
                             help='Posts are from same author (apply diversity penalty)')
    batch_parser.set_defaults(func=cmd_batch)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
