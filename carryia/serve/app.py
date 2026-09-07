"""P0-7 -- the Carryia app: a post-game "profile" view + conversational coach.

A wide profile screen over the committed snapshot: a profile header, a recent
Solo/Duo games rail, a performance KPI panel (placed against the Silver cohort),
and the **coach chat** (the grounded, cited RAG Q&A). Every number and every answer
runs on the same committed data + hybrid retriever: the P0-7 loaders and the
P0-6/P0-8 answer engine.

**The chat is conversational (P0-7).** History lives in session state and the whole
transcript renders; on each ask the earlier turns are prepended to the answer prompt
(`serve.rewrite.render_transcript`) so the coach has memory of the chat. A scoped
system addendum lets it answer questions ABOUT the conversation, while coaching advice
still stands solely on the retrieved, cited tips. Retrieval runs on the raw turn.

**Assets** are hotlinked from Data Dragon / Community Dragon at a pinned patch
(`ASSET_VERSION`). A live Streamlit page runs in the browser with no CSP, so
hotlinking is fine; the patch is pinned so committed art can't drift.

**Header fields (icon, level, rank, split win/loss)** are NOT in the match snapshot
-- they come from Summoner-V4 / League-V4 (platform route). A one-time author pull
(`carryia/personal/profile_pull.py`) captures them into `data/snapshot/profile.json`,
kept separate from `meta.json` (the regional match provenance) per the two-plane
separation. The header reads that committed file; the **win rate is the current
split, not lifetime** -- Riot exposes no career total.

Run:  streamlit run carryia/serve/app.py
"""

from __future__ import annotations

import base64
import json
import math
from datetime import date
from statistics import NormalDist

import streamlit as st
from dotenv import load_dotenv

from carryia.paths import DATA, ROOT
from carryia.personal import ingest as personal_ingest
from carryia.personal.metrics import player_metrics
from carryia.personal.producer import (
    load_cohort,
    panel_rows,
    place_metrics,
    render_context_block,
)
from carryia.pipeline.ingest import load_corpus
from carryia.serve import db
from carryia.serve.llm_backend import backend, make_client, model_id
from carryia.serve.monitoring import RAGWithMetrics
from carryia.serve.rag_helper import RAGBase
from carryia.serve.retrieval import build_hybrid_docs_retriever

load_dotenv()  # reviewer's .env for a local `streamlit run` (no-op under compose)

# --- asset resolution (pinned patch) -----------------------------------------

ASSET_VERSION = "16.17.1"
_DDRAGON = "https://ddragon.leagueoflegends.com/cdn"
_CDRAGON = "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-static-assets/global/default"

# Match-V5 championName is *mostly* the ddragon key, but a few diverge. Our subject
# pool (Seraphine/Lux/Neeko/Milio) is clean; the fixup keeps the resolver honest for
# the rest.
_CHAMP_KEY_FIXUP = {"Wukong": "MonkeyKing", "Fiddlesticks": "FiddleSticks"}


def _champ_key(name: str) -> str:
    return _CHAMP_KEY_FIXUP.get(name, name)


def champ_square(name: str) -> str:
    return f"{_DDRAGON}/{ASSET_VERSION}/img/champion/{_champ_key(name)}.png"


def champ_splash(name: str) -> str:
    # Evergreen (no patch in path); base skin `_0`. 1215x717 landscape -> right banner.
    return f"{_DDRAGON}/img/champion/splash/{_champ_key(name)}_0.jpg"


def profile_icon(icon_id: int) -> str:
    return f"{_DDRAGON}/{ASSET_VERSION}/img/profileicon/{icon_id}.png"


def rank_emblem(tier: str) -> str:
    return f"{_CDRAGON}/images/ranked-emblem/emblem-{tier.lower()}.png"


# --- subject profile (platform-route fields; see module docstring) ------------

@st.cache_data(show_spinner=False)
def _profile() -> dict:
    """The committed `profile.json` -- icon, level, rank, and split win/loss.
    Written once by `profile_pull.py`; reviewers read it, never re-pull. Rank fields
    are null when the subject is unranked in solo/duo this split."""
    return json.loads((DATA / "snapshot" / "profile.json").read_text())

# Layout: on non-mobile the page is locked to the viewport (no scroll) and the three
# panels fit snugly with aligned bottoms. The rail's height (`--panel-h`, a viewport
# calc in `_css`) drives the row; Performance + Coach use height="stretch" to match it,
# Coach filling the taller full column. Tune margins via `--vpad`/`--hdr` in `_css`.

# Empty-state prompts for the coach chat: label (shown) -> question (sent).
SUGGESTIONS = {
    "Why do I keep losing lane?":
        "Why do I keep losing lane as a support?",
    "Use my vision better":
        "How do I use my vision and warding better?",
    "What should I fix first?":
        "Based on my recent games, what is the single biggest thing I should fix first?",
}


# --- data --------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load():
    """Committed snapshot -> games (newest first), subject metrics, cohort, the
    judged / descriptive panel rows, meta, and the personal-context block the coach
    grounds on. Deterministic, keyless (P0-10)."""
    games = personal_ingest.load_snapshot()
    games = sorted(games, key=lambda g: g.game_creation, reverse=True)
    subject = player_metrics(games)
    cohort = load_cohort()
    judged, descriptive = panel_rows(subject, cohort)
    meta = json.loads((DATA / "snapshot" / "meta.json").read_text())
    block = render_context_block(
        place_metrics(subject, cohort),
        subject["games"],
        subject["whole_game"],
        cohort.get("meta", {}).get("player_count"),
    )
    return games, subject, cohort, judged, descriptive, meta, block


# --- coach engine ------------------------------------------------------------

@st.cache_resource(show_spinner="Building the coaching index (first load, ~10s)…")
def _retriever():
    """Hybrid retriever (keyword + vector, RRF-fused -- the P0-5 winner) over the
    committed tips. Once per process -- keyless, local embeddings, deterministic."""
    return build_hybrid_docs_retriever(load_corpus())


@st.cache_resource(show_spinner=False)
def _db_conn():
    """Monitoring DB connection once per process, or None if Postgres isn't up (a bare
    `streamlit run` still answers; compose brings Postgres so every ask is recorded)."""
    try:
        return db.connect()
    except Exception:
        return None


def _coach(conn):
    """Answer generator: hybrid retriever + the reviewer's LLM (P0-6 WHY default).
    Instrumented variant when the DB is up (P0-8), plain otherwise. Built on first ask
    so a missing key fails only there, with guidance -- not at page load."""
    common = dict(index=None, llm_client=make_client(), retriever=_retriever(),
                  model=model_id())
    if conn is not None:
        return RAGWithMetrics(store=lambda rec: db.save_conversation(conn, rec), **common) # type: ignore
    return RAGBase(**common) # type: ignore


def _ago(d: date) -> str:
    days = (date.today() - d).days
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    if days < 14:
        return f"{days} days ago"
    return d.strftime("%b %d")


# --- pieces ------------------------------------------------------------------

def _css() -> None:
    st.html(
        """
        <style>
          .block-container { padding-top: 2.2rem; max-width: 3000px; }
          .rail-title { font-weight: 600; opacity: .85; margin: 0 0 .7rem .1rem; }
          /* Top bar: a fake (decorative) "profile search" box + a "Demo mode" caution badge,
             side by side. Rendered as ONE st.html flex block so the layout is controlled
             entirely HERE -- not via Streamlit's horizontal-container / widget DOM, which would
             not lay the two out beside each other. Both are non-functional, purely for UI
             completion. The magnifier + caution triangle are base64 data-URI images (base64 so
             no raw-svg angle brackets trip the st.html sanitizer). The row height is reserved via
             --searchh so the viewport-fit budget (--panel-h) still sums to 100vh, bottoms aligned. */
          .topbar { display: flex; align-items: center; gap: .6rem; margin-bottom: .2rem; }
          .searchbox { display: flex; align-items: center; gap: .5rem; flex: 0 1 30rem;
            min-width: 0; padding: .5rem .75rem; border-radius: 6px;
            background: #12161F; border: 1px solid #625133; cursor: not-allowed; }
          .search-ic { width: 1rem; height: 1rem; flex: none; opacity: .85; }
          .search-ph { color: #E8E6E3; opacity: .5; font-size: .9rem; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; }
          /* "Demo mode" caution badge -- directly beside the search box (hugs its content).
             Amber text + triangle on a translucent amber fill with a muted-gold hairline. */
          .demo-badge { display: inline-flex; align-items: center; gap: .4rem; flex: none;
            padding: .3rem .6rem; border-radius: 6px; white-space: nowrap;
            background: rgba(224,168,61,.12); border: 1px solid #625133;
            color: #e0a83d; font-size: .78rem; font-weight: 600; letter-spacing: .2px; }
          .demo-ic { width: .95rem; height: .95rem; flex: none; }
          /* match-history cards -- champion splash bleeds in from the right */
          .mh-list { display: flex; flex-direction: column; gap: .35rem; flex: 1 1 auto; min-height: 0; }
          .mh-card { position: relative; overflow: hidden; flex: 1 1 0; min-height: 46px;
            border-radius: 0px; background: #12161F; display: flex; align-items: stretch;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.05); }
          .mh-art { position: absolute; inset: 0 0 0 auto; width: 62%; z-index: 0;
            background-size: cover; background-position: 78% 22%;
            /* Tame the splash: history is the least important content but the loudest, so
               desaturate + dim it at rest and restore full colour on hover. */
            filter: saturate(.95) brightness(.97); transition: filter .25s ease;
            -webkit-mask-image: linear-gradient(to right, transparent 0%, #000 18%);
                    mask-image: linear-gradient(to right, transparent 0%, #000 18%); }
          .mh-card:hover .mh-art { filter: saturate(1) brightness(1); }
          .mh-card::before { content: ""; position: absolute; inset: 0; z-index: 1;
            background: linear-gradient(to right,
              #12161F 40%, rgba(18,22,31,0) 54%); }
          .mh-bar { width: 5px; flex: none; z-index: 2; border-radius: 0px 0 0 0px; }
          .mh-card.win  .mh-bar { background: #3fb950; box-shadow: 0 0 10px 0px rgba(63,185,80,.55); }
          .mh-card.loss .mh-bar { background: #f0506e; box-shadow: 0 0 10px 0px rgba(240,80,110,.55); }
          .mh-body { position: relative; z-index: 2; padding: .35rem .8rem;
            display: flex; flex-direction: column; justify-content: center; gap: .08rem; }
          .mh-champ { font-weight: 700; font-size: 1rem; letter-spacing: .2px; }
          .mh-kda { font-size: .84rem; opacity: .92; }
          .mh-kda .win  { color: #4ec76a; } .mh-kda .loss { color: #f26d85; }
          .mh-meta { font-size: .75rem; opacity: .5; }

          /* Coach chat: recolour the bordered-container border to muted gold + square
             the corners. In 1.63 border=True draws the 1px border AND the radius on the
             inner block that carries the key class (.st-key-coach) -- NOT the wrapper
             (stLayoutWrapper has border-width:0). */
          .st-key-coach {
            border-radius: 0 !important;
            /* No left/right padding: the recessed chat screen bleeds full-width to the
               inner edge of the bronze frame, so the ONLY internal seams are the screen's
               top + bottom bezels (between the wordmark plate and the input console). The
               wordmark is centred and the input carries its own box, so neither needs the
               side inset; message text gets its breathing room from stChatMessage padding. */
            padding-left: 0 !important;
            padding-right: 0 !important;
            /* Bottom padding zeroed + position:relative so the message screen fills to the
               frame and the chat input can float absolutely INSIDE it (see input rule). */
            padding-bottom: 0 !important;
            position: relative !important;
            /* Zero the flex gap between the wordmark band and the screen: the band centres the
               logo within itself, but a gap BELOW it (with none above) pushed the logo ~7px high
               of the plate (top frame -> bezel). Removing it drops the top bezel to the band's
               bottom, so the logo -- unmoved, keeping its profile-row alignment -- is now centred
               vertically in the plate. */
            gap: 0 !important;
            /* Thick RUSTY-BRONZE MECHA frame (League/Arcane, Piltdown-brass feel). A flat
               1px gold hairline reads as a web border, not forged metal -- so the border is
               thickened to 3px and painted with a diagonal metallic GRADIENT via border-image
               (light from top-left): dark umber -> bronze -> a brass highlight band -> bronze
               -> rusted dark, so the frame catches light like a bevelled casting. border-image
               ignores radius, but corners are already square. Seated with a dark inner hairline
               (inset box-shadow) so the metal sits in the surface rather than floating. */
            border-width: 1px !important;
            border-style: solid !important;
            border-image: linear-gradient(135deg,
              #3a2818 0%, #6e4a2a 20%, #a07a44 40%, #c19a5c 50%,
              #96683a 62%, #6e4a2a 82%, #3a2818 100%) 1 !important;
            /* "Pop out" of the flat page: the coach is the only ELEVATED surface. On a
               near-black ground a black drop shadow can't read, so elevation is carried the
               dark-UI way -- a LIGHTER surface (#1A212F, a step above the flat #12161F cards)
               that looks nearer the light -- reinforced by a layered drop shadow (ambient +
               contact) and a faint top rim (inset light hairline). Every other panel keeps the
               flat #12161F fill and no shadow. The chatlog scroll-fade below fades into this
               same #1A212F so it doesn't band at the edges. */
            background: #1A212F !important;
            box-shadow:
              0 24px 50px -12px rgba(0,0,0,.9),
              0 8px 18px -6px rgba(0,0,0,.7),
              inset 0 0 0 1px rgba(0,0,0,.45),
              inset 0 2px 0 rgba(255,255,255,.06) !important;
          }
          /* Coach title is the Carryia wordmark (assets/carryia-logo.png), centred via the
             horizontal container. Width scales with the SCREEN via a viewport unit (13vw), not
             just the responsive type -- rem tracks the vmin font (i.e. height), so a fixed rem
             wouldn't fill more of a wider box. A % of the container resolves against an
             unstable containing block (the centred flex band shrink-wraps toward the logo) and
             stalls on wide screens; vw is container-independent and tracks width cleanly.
             clamp() bounds it: never below 11rem (readable on a narrow/mobile column) nor above
             22rem (so it doesn't sprawl on ultrawides).
             VERTICAL ALIGN WITH THE PROFILE ICON: the band is given the profile-icon height
             (6.4rem) + align-items:center so the logo centres in an equal-height box, BUT the
             coach panel insets its content by its own top border + padding, so that box starts
             ~1rem+1px below the profile row. The coach padding-top computes to (1rem - 1px) at
             every size and the border is 2px, so the inset is exactly 1rem+1px; a matching
             negative margin-top pulls the band's top up to the coach top (= the profile row's
             top), so the two equal-height boxes share a centre line at every viewport.
             Uses MIN-height, not height -- CSS height on a Streamlit container is ignored;
             min-height is honoured. */
          .st-key-coachlogo { min-height: 6.4rem !important; align-items: center !important;
                              margin-top: calc(-1rem - 1px) !important; }
          .st-key-coachlogo img { width: clamp(7rem, 8.5vw, 13rem) !important; height: auto !important; }
          /* Message rows. (1) BREATHING ROOM at the right: the coach panel has
             padding-right:0 (the screen bleeds full-width to the bronze frame) and
             Streamlit zeroes the right padding on transparent (assistant) messages, so a
             long cited answer ran flush against the frame. A rem right inset (scales with
             the vmin type) pulls the text off the edge. (2) NO USER BUBBLE: Streamlit fills
             the "user" role with a secondary-bg bubble (rgba(38,39,48,.5) on the
             stChatMessage node) -- dropped so the user turn reads as plain text
             and only the assistant answer carries any surface. User rows are keyed by their
             avatar: the subject's ddragon `profileicon` URL (the assistant's is a local
             /media asset), a direct image child of the message row. */
          .st-key-chatlog [data-testid="stChatMessage"] { padding-right: 1.4rem !important; }
          .st-key-chatlog [data-testid="stChatMessage"]:has(> img[src*="profileicon"]) {
            background: transparent !important;
          }
          /* Coach chat input: FLOATS inside the message screen. Its element-container is lifted
             out of the coach flex column (position:absolute against .st-key-coach, which is
             position:relative) and anchored to the bottom-centre, overlapping the bottom of the
             recessed screen -- so the input is a floating box INSIDE the messages, not a row in a
             separate band below them. Compact centred width (max-width). The chatlog's bottom
             padding (84px) keeps the last message / suggestions above it. z-index over the screen. */
          [data-testid="stElementContainer"]:has(> [data-testid="stChatInput"]) {
            position: absolute !important;
            left: 0; right: 0; bottom: 16px;
            margin: 0 auto !important;
            max-width: 95% !important;
            z-index: 5 !important;
          }
          [data-testid="stChatInput"] > div:first-child {
            background: #0A0E14 !important;
            border-color: #625133 !important;
            border-radius: 0 !important;
            /* Match the top search box's height. .searchbox is .5rem/.75rem padding + a
               .9rem line + 1px border; Streamlit's default chat input renders shorter, so
               pin the same box height here and centre the textarea in it. Tune --chatinputh
               if the two boxes drift. */
            min-height: var(--chatinputh, 2.75rem) !important;
            display: flex !important;
            align-items: center !important;
          }
          /* Drop Streamlit's own textarea min-height so the box height above (not the
             textarea's default) sets the size, and the single line centres. */
          [data-testid="stChatInput"] textarea {
            min-height: 0 !important;
          }
          /* Focus affordance: pinning the border above killed the default gold flip, so
             focus had no signal. Add a gold accent ring + faint bloom on :focus-within --
             the same box-shadow glow idiom as the match-card win/loss bars. box-shadow
             (not border-width) so it doesn't nudge layout by 1px. */
          [data-testid="stChatInput"]:focus-within > div:first-child {
            box-shadow: 0 0 0 1px rgba(200,170,110,.5), 0 0 6px rgba(200,170,110,.15) !important;
          }
          /* No send button -- Enter submits (submit_mode="disable"); the up-arrow glyph
             was visual noise. */
          [data-testid="stChatInputSubmitButton"] { display: none !important; }
          /* Empty-state suggestions: small, rounded, right-aligned "floating" chips at the
             bottom right -- suggestive, not imposing. Buttons are width="content" so they hug
             their text; the column aligns them to the right (align-items:flex-end) and each
             stButton wrapper right-aligns its inline-block button (text-align:right).
             padding-right matches the floating input's right gap ((100% - 95% max-width)/2 =
             2.5%) so the chips' right edge lines up with the input box's right edge. */
          .st-key-suggestions {
            align-items: flex-end !important;
            gap: .35rem !important;
            padding-right: 2.5% !important;
          }
          .st-key-suggestions [data-testid="stButton"] { text-align: right !important; }
          .st-key-suggestions button {
            width: auto !important;
            border-radius: 8px !important;
            min-height: 0 !important;
            padding: .2rem .65rem !important;
            font-size: .78rem !important;
            line-height: 1.35 !important;
            background: rgba(18,22,31,.6) !important;   /* card fill, translucent -> floats */
            border-color: #625133 !important;           /* muted-gold hairline, ties to coach */
            color: #E8E6E3 !important;
            box-shadow: 0 1px 4px rgba(0,0,0,.3) !important;
            opacity: .8;
          }
          .st-key-suggestions button:hover {
            opacity: 1 !important; border-color: #C8AA6E !important;
          }
          /* Pin the suggestions to the BOTTOM of the stretch chatlog (just above the input):
             the chatlog is a flex column, so margin-top:auto on its child that holds the
             suggestions pushes it down past the empty space. */
          .st-key-chatlog > *:has(.st-key-suggestions) { margin-top: auto !important; }
          /* Resting-state intro: sits at the TOP of the empty chatlog (below the wordmark)
             so the screen carries content -- a blurb + two real data points -- rather than
             reading as a void until the first question. */
          .intro { padding: 1.2rem 1.2rem .3rem; }
          .intro-lead { font-size: 1.05rem; font-weight: 700; color: #E8E6E3; }
          .intro-sub { font-size: .82rem; opacity: .6; line-height: 1.4; margin-top: .3rem; }
          .intro-hl { display: flex; flex-direction: column; gap: .4rem; margin-top: .9rem; }
          .intro-chip { font-size: .82rem; font-weight: 600; padding: .4rem .6rem;
            background: rgba(18,22,31,.6); box-shadow: inset 0 0 0 1px #625133;
            border-left: 3px solid #625133; }
          .intro-chip span { display: block; font-size: .66rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: .5px; opacity: .55; margin-bottom: .1rem; }
          .intro-chip.up   { border-left-color: #4ec76a; }
          .intro-chip.down { border-left-color: #f26d85; }

          /* profile win-rate: a right-aligned header stat (far right, after the rank),
             reusing the match-card win/loss palette so header and rail read as one.
             All rem, so it scales. */
          /* The big % is coloured League gold (#C8AA6E, the app accent) rather than the loud
             match-card green/red -- on-palette, so it reads as a highlighted stat in the app's
             own vibe instead of shouting. The up/down class stays on the node (harmless) but no
             longer switches colour; the small W/L record below still carries the win/loss cue. */
          .wr { text-align: right; line-height: 1.25; }
          .wr-pct { font-size: 1.8rem; font-weight: 600; color: #C8AA6E; }
          .wr-cap { font-size: .8rem; font-weight: 500; opacity: .65; margin-left: .2rem; }
          .wr-rec { font-size: .95rem; font-weight: 500; }
          .wr-rec .win  { color: #4ec76a; } .wr-rec .loss { color: #f26d85; }
          .wr-lbl { font-weight: 400; opacity: .5; }

          /* --- performance panel: three-tier stats ------
             Bell = cohort comparison, card = no comparison; the bell fades out
             top->bottom. One bell component, reused small+static in tier 2. */
          /* Tighten the perf stack: many tiers -> the default ~1rem inter-element gap
             alone overflowed --panel-h at 1366. Also trim the tab component's own
             top/bottom padding. */
          .st-key-perf { gap: .25rem !important; }
          .st-key-perf [data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: .3rem !important; }
          /* Phase tabs span the full panel width: the tablist stretches and each tab takes an
             equal share, label centred. (Recent Streamlit uses react-aria: `[role=tablist]` +
             `[data-testid=stTab]`, not the old baseweb DOM.) */
          .st-key-perf [role="tablist"] { display: flex !important; width: 100% !important; }
          .st-key-perf [data-testid="stTab"] { flex: 1 1 0 !important; justify-content: center !important; }
          /* Active tab gets a translucent gold FILL (ties to the bell's shade colour) on top of the
             default gold underline, so the selected phase reads at a glance. */
          .st-key-perf [data-testid="stTab"][aria-selected="true"] {
            background: rgba(200,170,110,.14) !important; }
          /* Fill the panel height. The perf stack is a flex column stretched to --panel-h,
             but every tier is flex-grow:0, so the tiers pack to the top and the slack pools
             at the bottom. Two flex spacers (.perf-gap) sit in the seams -- after tier 1 and
             after tier 2 -- and absorb the slack equally, so the three tiers spread across the
             full height with tier 3 bottom-aligned to the rail + coach. A spacer collapses to
             0 when there's no slack (mobile, where the panel sizes to content), so it's inert
             there -- no need to gate it behind the media query. */
          [data-testid="stElementContainer"]:has(.perf-gap) { flex: 1 1 0 !important; }
          .tier-h { font-weight: 600; opacity: .85; font-size: 1rem; margin: .3rem 0 .35rem; }
          /* Marker-colour key, overlaid in the bell's top-right corner (curve is empty there).
             Faint pill so it stays legible if it ever grazes the curve. */
          .bell-legend { position: absolute; top: .1rem; right: .3rem; z-index: 2;
            display: flex; flex-direction: column; align-items: flex-start; gap: .18rem;
            font-size: .62rem; font-weight: 600; line-height: 1; padding: .28rem .5rem;
            border-radius: 3px; background: rgba(10,14,20,.5); }
          .bell-legend .bl-lead { opacity: .5; font-weight: 500; margin-bottom: .08rem; }
          .bell-legend .bl-row { display: flex; align-items: center; gap: .35rem; }
          .bell-legend .bl-ln { display: inline-block; flex: none; width: 3px; height: .7rem;
            border-radius: 1px; }
          .bell-img { width: 100%; height: auto; display: block; }
          /* Full-width bell with the legend as a compact row BENEATH it (was side-by-side).
             The shared bell uses a wide/flat viewBox so spanning full width doesn't blow the
             height budget at 1366 (see _performance). */
          .bellrow { display: flex; flex-direction: column; gap: .4rem; }
          /* The shared bell + its metric glyphs: each .bell-ic is absolutely placed under
             its marker line (left% = the marker's percentile-x), so the bell itself keys
             which curve line is which metric. */
          .bell-wrap { position: relative; }
          /* Anchor each glyph to the bell's BASELINE (where the marker lines end), not the
             SVG's bottom edge. The svg has a foot=20 band below the baseline in a vb_h=96 box,
             so the baseline sits 20/96 = 20.8% up from the bottom; that band scales with the
             SVG, so pinning to bottom:0 left a big gap under the curve on wide screens. Sitting
             the glyph top ~just under the baseline keeps it hugging the curve at every size. */
          .bell-ic { position: absolute; bottom: calc(20.8% - 1.5rem); transform: translateX(-50%);
            height: 1.1rem; width: auto; opacity: .9; }
          /* Phased metric cards beneath the bell: one per marker, carrying the raw numeric
             value the bell can't show (position there is percentile only). */
          .pcards { display: flex; flex-direction: row; gap: .35rem; }
          .pcard { flex: 1 1 0; min-width: 0; display: flex; align-items: center;
            justify-content: space-between; gap: .4rem; background: #12161F; border-radius: 0;
            padding: .4rem .55rem .45rem; box-shadow: inset 0 0 0 1px #625133; }
          .pcard-body { min-width: 0; flex: 1; }
          .pcard-title { font-size: .72rem; opacity: .8; line-height: 1.2; }
          .pcard-val { font-size: 1.35rem; font-weight: 700; line-height: 1; margin-top: .12rem; }
          .pcard-cap { font-size: .68rem; opacity: .6; margin-top: .15rem; }
          .pcard-pct { font-size: 1.35rem; font-weight: 700; opacity: .95; flex: none;
            display: flex; flex-direction: column; align-items: flex-end; line-height: 1; }
          /* "percentile" caption under the ranking ordinal. */
          .pct-lbl { font-size: .6rem; font-weight: 600; opacity: .5; text-transform: uppercase;
            letter-spacing: .4px; margin-top: .15rem; }
          /* Metric glyph on the left of the Tier 2/3 metric cards. (Tier 1 keys its
             metrics via the bell marker glyphs instead, so its cards carry no icon.) */
          .card-ic { flex: none; width: 1.6rem; height: 1.6rem; opacity: 1; }
          .bcard { display: flex; align-items: flex-start; gap: .55rem; background: #12161F;
            border-radius: 0; padding: .5rem .7rem .55rem; box-shadow: inset 0 0 0 1px #625133; }
          .bcard .card-ic { margin-top: .1rem; }
          /* Tier 2: the bell sits ABOVE its card (outside it), in a `.bwrap` column that stacks
             bell over card. Sized by height + centred, held to a moderate size. */
          .bwrap { display: flex; flex-direction: column; }
          .bwrap .bell-img { width: auto; max-height: 7rem; margin: 0 auto .45rem; }
          .bcard-body { flex: 1; min-width: 0; }
          .bcard-title { font-size: .8rem; opacity: .8; }
          .bcard-valrow { display: flex; align-items: baseline; justify-content: space-between;
            gap: .4rem; }
          /* Value + cohort as their OWN column, mirroring .bcard-pct (ordinal + label), so the
             value->cohort gap equals the ordinal->percentile gap. (Without this, the cohort sat
             below the whole row, which the two-line pct block stretched tall -> a wider gap.) */
          .bcard-valcol { display: flex; flex-direction: column; align-items: flex-start; }
          .bcard-pct { font-size: 1.6rem; font-weight: 700; opacity: .95; flex: none;
            display: flex; flex-direction: column; align-items: flex-end; line-height: 1; }
          .bcard-val { font-size: 1.6rem; font-weight: 700; line-height: 1; margin-top: .12rem; }
          .bcard-cap { font-size: .72rem; opacity: .6; margin-top: .15rem; margin-bottom: .3rem; }
          .scards { display: flex; gap: .35rem; flex-wrap: wrap; }
          .scard { flex: 1 1 0; min-width: 82px; display: flex; align-items: center; gap: .5rem;
            background: #12161F; border-radius: 0; padding: .45rem .6rem;
            box-shadow: inset 0 0 0 1px #625133; }
          .scard-body { min-width: 0; flex: 1; }
          .scard-val { font-size: 1.25rem; font-weight: 700; line-height: 1.15; }
          .scard-lbl { font-size: .7rem; opacity: .55; margin-bottom: .1rem; }

          /* profile name + level: ONE st.html block, so it's a clean box that the column's
             vertical_alignment centres exactly on the icon. (Streamlit's h2 heading uses
             internal negative margins + reserved padding, so the two-markdown version's box
             never matched its visible content and mis-centred / overlapped.) Name font in rem
             so it scales like the old h2 did. */
          .pname { display: flex; flex-direction: column; align-items: flex-start; gap: .45rem; }
          .pname-name { font-size: 2.7rem; font-weight: 700; line-height: 1.1; letter-spacing: .2px;
            white-space: nowrap; }  /* never wrap to 2 lines -- that would grow the header past --hdr */
          .pname-tag { opacity: .45; }
          .pname-lvl { font-size: 1.2rem; font-weight: 600; color: #3fb950; }

          /* Icons hug their labels: an icon sits at the LEFT of its column while its
             label starts at the left of the NEXT column, so the column's trailing
             whitespace reads as a gap. st.image(width=..) makes the image block only as
             wide as the icon; margin-left:auto right-aligns that block within its column
             so the profile icon sits next to the name (and the emblem next to the
             tier/LP). Scales at every font size. */
          .st-key-pfp [data-testid="stElementContainer"],
          .st-key-rank [data-testid="stElementContainer"] { margin-left: auto; }

          /* rail wrapper: cards flex to fill the panel height, so the list never scrolls */
          .mh-wrap { display: flex; flex-direction: column; height: 700px; }

          /* --- viewport-fit: on non-mobile the whole page fits; nothing scrolls --- */
          :root {
            /* Streamlit's default top bar is a FIXED ~56px (it doesn't scale with our
               html font-size), so --hbar is px, not rem -- the whole app is pushed down
               by exactly the bar's height to sit below it, and that same height comes out
               of the 100vh panel budget so nothing overflows. */
            --hbar: 56px;     /* Streamlit default header bar -- app clears it */
            --vpad: 4.4rem;   /* symmetric top + bottom margin (2.2rem each) */
            --hdr: 9.8rem;    /* profile header height (main column starts this far below) */
            --searchh: 3.8rem; /* reserved height of the disabled top search row (input + its
                                  element gap); kept OUT of the panel budget so the 100vh fit
                                  and aligned bottoms still hold. Biased slightly HIGH -- under-
                                  reserving clips the panels (stMain is overflow:hidden), over-
                                  reserving just leaves a small bottom gap. Tune if bottoms drift. */
            --panel-h: calc(100vh - var(--hbar) - var(--vpad) - var(--hdr) - var(--searchh));
          }
          @media (min-width: 641px) {
            /* Responsive type: scales with the smaller viewport side (vmin) so text
               tracks screen size without overflowing the height-locked panels. All
               Streamlit text and the rem-based offsets below scale with it. */
            html { font-size: clamp(13px, 1.5vmin, 20px); }
            /* Icons in rem so the header scales uniformly with the type. Profile is a
               square 300x300 -> size by width; the rank emblem is a 16:9 crest with side
               padding -> size by height (width auto) or it renders squashed and tiny. */
            .st-key-pfp img  { width: 6.4rem !important; height: auto !important; }
            /* Emblem crest is only ~20%x25% of its 1280x720 canvas (rest is transparent
               padding), so fitting the whole frame renders it tiny. object-view-box crops
               to the crest region first. Size the CROP by WIDTH (rem, scales with the icon)
               with height auto so the ~1.5:1 aspect is preserved -- NOT by height with
               max-width:none, which let the wide crest overflow its column and overlap the
               tier/LP label once the font kept growing past the 1500px-capped container
               (big/tall external monitors, where columns stop widening but rem doesn't).
               max-width:100% is the hard stop: on those screens the emblem shrinks to fit
               rather than bleeding over the label. 7.3rem wide -> ~4.9rem tall (matches the
               profile icon at normal sizes). */
            .st-key-rank img { object-view-box: inset(31% 37% 38% 37%) !important;
                               width: 9.5rem !important; height: auto !important;
                               max-width: 100% !important; }
            [data-testid="stMain"] { overflow: hidden; }
            /* Make space for Streamlit's default header bar: instead of hiding it (the
               old transparent-header hack that let content bleed underneath), start the
               whole app --hbar below the top. padding-top = the bar's height + the usual
               2.2rem margin; --panel-h subtracts --hbar too, so the page still fits 100vh. */
            .block-container { height: 100vh; padding-top: calc(var(--hbar) + 2.2rem);
                               padding-bottom: .6rem; }
            /* The rail height drives the row; Performance + Coach use height="stretch"
               (Streamlit's own flex fill) to match it -- Coach fills the taller full column. */
            .mh-wrap { height: var(--panel-h) !important; }

            /* Coach chat log scrolls INSIDE instead of growing the page. height="stretch"
               fills the column but does NOT cap the log: a long history grows the column, the
               row, and the whole page. The exact failure: the columns row (stHorizontalBlock)
               DOES stay bounded, but the chat column
               (stColumn) grows to its content -- align-items:stretch does not cap it, and its
               display:block default makes the child's flex-grow/min-height inert. So the fix is
               a chain, every link needed:
                 (a) block-container -> flex column, so the row gets a definite (100vh-bounded)
                     height instead of sizing to content;
                 (b) chat column -> flex column AND height:100%, pinning it to that definite row
                     height (this is THE cap -- stretch/min-height alone never capped it);
                 (c) min-height:0 down the whole chain, so each flex ancestor may shrink below
                     its content (default min-height:auto pins it to content and defeats the cap);
                 (d) the log -> overflow-y:auto to scroll, and its direct children ->
                     flex-shrink:0 so they keep natural height and overflow instead of squishing
                     (the log is a flex column; without this the messages compress and never
                     scroll).
               Non-mobile only (block-container is 100vh-bounded here); mobile flows normally. */
            .block-container:has(.st-key-chatlog),
            [data-testid="stColumn"]:has(.st-key-chatlog) { display: flex; flex-direction: column; }
            [data-testid="stColumn"]:has(.st-key-chatlog) { height: 100%; }
            .block-container *:has(.st-key-chatlog),
            .st-key-chatlog { min-height: 0; }
            .st-key-chatlog { overflow-y: auto; }
            /* Hide the scrollbar chrome but keep scrolling: scrollbar-width:none (Firefox/
               standard) + the webkit pseudo (Chromium/Safari). */
            .st-key-chatlog { scrollbar-width: none; }
            .st-key-chatlog::-webkit-scrollbar { display: none; }
            /* RECESSED inner "screen": the message log is set INTO the raised console. A
               darker fill (#12161F, a step below the #1A212F panel surface) + a top+bottom
               inset shadow read as a display recessed behind a bezel (no left/right edge --
               the screen runs full-width to the frame) -- so the wordmark plate
               above and the input console below (both on the lighter #1A212F surface) lift
               away from the messages. Bronze bezel hairlines (top + bottom) mark the two
               seams and tie to the frame. Tonal ladder: input well #0A0E14 (deepest) <
               message screen #12161F < panel surface #1A212F (raised).
               Scroll-fade indicators (top + bottom), pure CSS (Lea Verou layered-gradient --
               st.html strips JS anyway): two COVER gradients (attachment:local, scroll WITH
               content) fading to the screen fill #12161F, over two SHADOW gradients fixed to
               the edges (attachment:scroll). A cover overlaps its shadow only at that extreme
               -> top shadow hidden at scrollTop 0, bottom shadow hidden at max scroll, each
               shows once there's content past that edge. The final solid #12161F in the
               shorthand is the background-COLOUR behind the four image layers. */
            .st-key-chatlog {
              background:
                linear-gradient(#12161F 30%, rgba(18,22,31,0)) center top,
                linear-gradient(rgba(18,22,31,0), #12161F 70%) center bottom,
                radial-gradient(farthest-side at 50% 0, rgba(0,0,0,.65), rgba(0,0,0,0)) center top,
                radial-gradient(farthest-side at 50% 100%, rgba(0,0,0,.65), rgba(0,0,0,0)) center bottom,
                #12161F;
              background-repeat: no-repeat;
              background-size: 100% 34px, 100% 34px, 100% 24px, 100% 24px;
              background-attachment: local, local, scroll, scroll;
              /* Top bezel only: the wordmark plate still seams here. The BOTTOM bezel is gone
                 -- the screen now runs to the frame and the chat input floats INSIDE it (no
                 separate input band). padding-bottom keeps the last message / the suggestion
                 cluster clear of the floating input box. */
              border-top: 1px solid rgba(122,83,48,.55) !important;
              padding-bottom: calc(16px + 4.6rem) !important;   /* reserve = 16px input bottom-offset
                 + ~3.4rem input box (it SCALES with the vmin type: ~44px@13px root, ~67px@20px) +
                 ~1.2rem gap to the pinned suggestion chips. All-rem clearance so the gap stays
                 proportional -- a fixed px over-reserved on a short MacBook Air (input is only 44px
                 there) and left a huge gap. */
              box-shadow:
                inset 0 3px 7px -2px rgba(0,0,0,.6),
                inset 0 -3px 7px -2px rgba(0,0,0,.5) !important;
            }
            .st-key-chatlog > * { flex-shrink: 0; }
            /* Dim message text as it scrolls behind the floating input instead of letting it
               hard-disappear. The input box is opaque and floats over the chatlog's bottom band
               (see the float rule above), so a long answer's lines cut off sharply behind it.
               This foreground scrim over the coach's bottom band fades text into the screen fill
               (#12161F) as it enters the reserved zone, so it DIMS out. The scrim spans exactly
               the chatlog's reserved padding (16px + 4.6rem): opaque from the bottom up to the
               input's top (16px offset + ~3.4rem box), then fading to transparent over the ~1.2rem
               gap above it -- so the resting last line (which sits at 4.6rem) stays crisp while
               anything scrolling into the gap/behind the input dims. Stops in calc() lengths so it
               scales with the vmin type. z-index 4 = above messages, below the input (z-index 5)
               so the input stays crisp; pointer-events:none so clicks reach the messages. Gated on
               :has(a chat message) so the empty-state suggestion chips (no scroll) aren't dimmed. */
            .st-key-coach:has([data-testid="stChatMessage"])::after {
              content: "";
              position: absolute;
              left: 0; right: 0; bottom: 0;
              height: calc(16px + 4.6rem);
              background: linear-gradient(to top,
                #12161F 0,
                #12161F calc(16px + 3.4rem),
                rgba(18,22,31,0) calc(16px + 4.6rem));
              pointer-events: none;
              z-index: 4;
            }
            /* Pin the chat avatar while its message scrolls past (behaviour A). The
               avatar is the first flex child of stChatMessage; position:sticky sticks it
               to the top of the scroll box (.st-key-chatlog is the scroll ancestor),
               bounded by its own message row -- so a long answer's icon stays in view,
               then leaves with the message. top offset clears the top scroll-fade band. */
            .st-key-chatlog [data-testid="stChatMessage"] { align-items: flex-start; }
            .st-key-chatlog [data-testid="stChatMessage"] > img {
              position: sticky; top: .5rem; z-index: 3;
              /* Drop the avatar by the first line's line-height leading so its top lines up
                 with the text's cap height, not the (higher) empty top of the line box. */
              margin-top: .45rem;
            }
            /* The real gap: a Markdown heading (the coach's answers open on an H1) carries
               Streamlit's default 1.5rem (spacing.xl) top padding, so the text started far
               below the top-aligned avatar. Zero the top spacing on the FIRST block of any
               chat message so the first line begins at the row top, level with the avatar;
               spacing between later blocks is untouched. */
            .st-key-chatlog [data-testid="stChatMessage"]
              [data-testid="stMarkdownContainer"] > :first-child {
              margin-top: 0 !important;
              padding-top: 0 !important;
            }
          }
        </style>
        """
    )


def _win_rate_stat(wins: int, losses: int) -> str:
    """The far-right header stat: a big win-rate % (coloured across the 50% line),
    the record in the match-card win/loss palette, and a 'this split' qualifier --
    Riot exposes no lifetime rate (see module docstring). Right-aligned to the edge."""
    total = wins + losses
    wr = wins / total * 100 if total else 0.0
    trend = "up" if wr >= 50 else "down"
    return (
        "<div class='wr'>"
        f"<div class='wr-pct {trend}'>{wr:.1f}%<span class='wr-cap'>win rate</span></div>"
        f"<div class='wr-rec'><span class='win'>{wins}W</span> <span class='loss'>{losses}L</span>"
        f"<span class='wr-lbl'> · this split</span></div>"
        "</div>"
    )


def _header(name: str, tag: str, region: str, profile: dict) -> None:
    # icon | name/level (left) ......... rank | win-rate (a single right-aligned cluster).
    # The wide name column absorbs the slack so rank + win-rate sit together at the right
    # edge instead of floating as separate mid-header islands.
    hi, hname, hrank, hwr = st.columns([1, 4.1, 2.4, 1.9], vertical_alignment="center")
    with hi:
        with st.container(key="pfp"):
            st.image(profile_icon(profile["profile_icon_id"]), width=78)
    with hname:
        st.html(
            "<div class='pname'>"
            f"<div class='pname-name'>{name} <span class='pname-tag'>#{tag}</span></div>"
            f"<span class='pname-lvl'>Lvl {profile['summoner_level']}</span>"
            "</div>"
        )
    with hrank:
        if profile.get("tier"):
            # Emblem + tier/LP as ONE centred group, so the rank sits midway between the
            # profile block and the win-rate. A horizontal container centres the pair
            # within the column; nested st.columns pinned it to a sub-column edge (right
            # of centre) instead. The column ratios put the column centre on that midpoint.
            with st.container(horizontal=True, horizontal_alignment="right",
                              vertical_alignment="center", gap="xsmall"):
                with st.container(key="rank", width="content"):
                    st.image(rank_emblem(profile["tier"]), width=58)
                st.markdown(
                    f"**{profile['tier'].title()} {profile['rank']}**  \n"
                    f":gray[{profile['league_points']} LP]"
                )
        else:
            st.markdown("### :gray[Unranked]")
    with hwr:
        if profile.get("wins") is not None:  # null when unranked this split
            st.html(_win_rate_stat(profile["wins"], profile["losses"]))


def _recent_games(games) -> None:
    cards = []
    for g in games:
        res = "win" if g.win else "loss"
        dur = f"{g.game_duration_s // 60}:{g.game_duration_s % 60:02d}"
        cards.append(
            f"<div class='mh-card {res}'>"
            f"<div class='mh-bar'></div>"
            f"<div class='mh-body'>"
            f"<div class='mh-champ'>{g.champion}</div>"
            f"<div class='mh-kda'>{g.kills}/{g.deaths}/{g.assists} "
            f"<span class='{res}'>· {g.kda:.2f} KDA</span></div>"
            f"<div class='mh-meta'>{dur} · {_ago(g.game_creation)}</div>"
            f"</div>"
            f"<div class='mh-art' style=\"background-image:url('{champ_splash(g.champion)}')\"></div>"
            f"</div>"
        )
    st.html(
        "<div class='mh-wrap'>"
        "<div class='rail-title'>Recent Solo/Duo Games</div>"
        f"<div class='mh-list'>{''.join(cards)}</div>"
        "</div>"
    )


# --- performance panel: three-tier stats ------------------
# The visual FORM encodes the data type: a bell = a cohort comparison, a card =
# no comparison. Both cohort tiers reuse ONE bell component; the bell fades out
# top->bottom as comparison richness drops (phased bell -> bell-in-card -> plain
# card). Position on the bell is the RAW empirical percentile; marker COLOUR is the
# coaching standing (valence-aware), so `deaths` on the left reads as good (green)
# while a left-side vision marker reads as weak (red).

_STANDING_COLOR = {"weak": "#f26d85", "typical": "#C8AA6E", "strong": "#4ec76a"}

# Short labels for the bell legends / cards (the phase prefix is the tab, so it's
# dropped here). Keeps the app off producer's private _LABELS.
_STAT_LABELS = {
    "deaths": "Deaths",
    "ward_activity": "Ward activity",
    "objective_participation": "Objective participation",
    "vision_score_per_min": "Vision score / min",
    "kill_participation": "Kill participation",
}
_AS_PCT = frozenset({"kill_participation"})  # stored 0..1 -> shown as a percentage

# Tier 3 descriptive KPIs hidden from the UI cards (matched on their _LABELS text). Still
# flow into the coach's personal-context block -- this trims the panel, not the grounding.
_HIDDEN_DESCRIPTIVE = frozenset({"team damage %", "KDA"})

# Metric glyphs (assets/metric/*.svg) -- one per metric, shown on the left of every
# metric card (all three tiers) and as the phased-bell marker glyphs.
# Inlined as a base64 <img> -- st.html strips a raw inline <svg>.
_METRIC_ICONS = {
    "deaths": "metric-deaths.svg",
    "ward_activity": "metric-ward-activity.svg",
    "objective_participation": "metric-objective.svg",
    "vision_score_per_min": "metric-vision-min.svg",
    "kill_participation": "metric-kill-participation.svg",
    "effective_heal_shield": "metric-heal-shield.svg",
    "enemy_immobilizations": "metric-crowd-control.svg",
    "ward_takedowns": "metric-ward-takedowns.svg",
}
# Tier 3 descriptive cards carry the display label, not the metric key -- map back so
# they can look up the same glyph.
_DESC_METRIC = {
    "effective heal+shield": "effective_heal_shield",
    "crowd control": "enemy_immobilizations",
    "ward takedowns": "ward_takedowns",
}
# Prettified display labels for the Tier 3 cards (producer's descriptive labels are
# lowercase). Display-only -- the lowercase key still drives _DESC_METRIC/_HIDDEN_DESCRIPTIVE
# matching and the LLM context block, so this doesn't ripple past the UI.
_DESC_DISPLAY = {
    "effective heal+shield": "Effective Heal + Shield",
    "crowd control": "Crowd Control",
    "ward takedowns": "Ward Takedowns",
}


@st.cache_data
def _metric_icon_src(metric: str) -> str:
    """The metric glyph (assets/metric/*.svg) as a base64 data URI, for an <img src>."""
    svg = (ROOT / "assets" / "metric" / _METRIC_ICONS[metric]).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


def _val(metric: str, v: float) -> str:
    return f"{v * 100:.0f}%" if metric in _AS_PCT else f"{v:.1f}"


def _ord(pct: float) -> str:
    """Percentile as an ordinal: 38.8 -> '39th'."""
    n = round(pct)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_STD = NormalDist()  # standard normal -> percentile<->z (the symmetric-cohort assumption)

# The bell's own stylesheet: it renders inside an <img> data-URI (see _bell_svg), an
# isolated document the page CSS can't reach, so styling ships INSIDE the SVG. (st.html
# strips a raw inline <svg> -- a base64 <img> src survives it.)
_BELL_STYLE = (
    "<style>"
    ".ax{stroke:rgba(255,255,255,.14);stroke-width:1}"
    ".md{stroke:rgba(255,255,255,.18);stroke-width:1;stroke-dasharray:2 3}"
    ".cv{fill:none;stroke:#5b6472;stroke-width:1.6}"
    ".fl{fill:rgba(200,170,110,.16)}"
    ".ml{stroke-width:1.6;opacity:.9}"
    ".pt{stroke:#0A0E14;stroke-width:1.5}"
    "</style>"
)


def _bell_svg(markers, *, shade_pct=None, vb_w=260, vb_h=118) -> str:
    """A canonical standard-normal bell with a marker per (percentile, colour).

    The cohort is assumed symmetric, so the curve is the SAME
    fixed shape every time and the snapshot only needs a percentile per stat. A
    marker sits at the point on the curve whose LEFT-tail area equals its percentile
    (`inv_cdf`), so shading to its left is literally that percentile of the cohort.
    Returned as a base64 <img> (survives the st.html sanitizer); the viewBox +
    the .bell-img { width:100% } rule scale it to its box."""
    pad, top, foot = 16, 12, 20
    base = vb_h - foot
    amp = base - top
    zmax = 2.8

    def zx(z: float) -> float:
        return pad + (z + zmax) / (2 * zmax) * (vb_w - 2 * pad)

    def z_of(pct: float) -> float:
        p = min(max(pct, 0.5), 99.5) / 100  # clamp: inv_cdf(0)/(1) are infinite
        return max(-zmax, min(zmax, _STD.inv_cdf(p)))

    def cy(z: float) -> float:
        return base - math.exp(-z * z / 2) * amp

    n = 64
    zs = [-zmax + 2 * zmax * i / n for i in range(n + 1)]
    curve = "M " + " L ".join(f"{zx(z):.1f},{cy(z):.1f}" for z in zs)

    parts = [
        _BELL_STYLE,
        f"<line x1='{pad}' y1='{base}' x2='{vb_w - pad}' y2='{base}' class='ax'/>",
        f"<line x1='{zx(0):.1f}' y1='{cy(0):.1f}' x2='{zx(0):.1f}' y2='{base}' class='md'/>",
    ]
    if shade_pct is not None:  # single-marker tiers only: shade the percentile area
        zt = z_of(shade_pct)
        seg = [-zmax + (zt + zmax) * i / 40 for i in range(41)]
        area = (f"M {pad},{base} L "
                + " L ".join(f"{zx(z):.1f},{cy(z):.1f}" for z in seg)
                + f" L {zx(zt):.1f},{base} Z")
        parts.append(f"<path d='{area}' class='fl'/>")
    parts.append(f"<path d='{curve}' class='cv'/>")
    for pct, color in markers:
        z = z_of(pct)
        parts.append(
            f"<line x1='{zx(z):.1f}' y1='{cy(z):.1f}' x2='{zx(z):.1f}' y2='{base}' "
            f"stroke='{color}' class='ml'/>"
            f"<circle cx='{zx(z):.1f}' cy='{cy(z):.1f}' r='4' fill='{color}' class='pt'/>"
        )
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {vb_w} {vb_h}'>"
           f"{''.join(parts)}</svg>")
    b64 = base64.b64encode(svg.encode()).decode()
    return f"<img class='bell-img' src='data:image/svg+xml;base64,{b64}'/>"


def _bell_x_frac(pct: float, *, vb_w: int = 560, pad: int = 16, zmax: float = 2.8) -> float:
    """Left position (0..1 of viewBox width) of a percentile marker on the bell,
    matching _bell_svg's zx(z_of(pct)). Used to place the metric glyph under a marker."""
    p = min(max(pct, 0.5), 99.5) / 100
    z = max(-zmax, min(zmax, _STD.inv_cdf(p)))
    return (pad + (z + zmax) / (2 * zmax) * (vb_w - 2 * pad)) / vb_w


def _performance(subject, cohort, descriptive) -> None:
    placements = place_metrics(subject, cohort)
    with st.container(border=False, height="stretch", key="perf"):
        # Tier 1 -- By phase vs. Peers: one shared bell, tabbed BY PHASE (percentile
        # is a common axis, so different-unit metrics compare on one curve; a phase
        # tab reads as a profile of that moment of the game).
        st.html("<div class='tier-h'>By phase vs. Peers</div>")
        # Marker-colour key, moved OUT of a header note and INTO the bell's top-right corner
        # (empty space there -- the curve tapers to the baseline at the edges). One per tab,
        # but only the active tab renders, so it reads as a single in-graph legend.
        legend = (
            "<div class='bell-legend'><span class='bl-lead'>your standing</span>"
            "<span class='bl-row'><i class='bl-ln' style='background:#f26d85'></i>weak</span>"
            "<span class='bl-row'><i class='bl-ln' style='background:#C8AA6E'></i>typical</span>"
            "<span class='bl-row'><i class='bl-ln' style='background:#4ec76a'></i>strong</span>"
            "</div>"
        )
        phased = [p for p in placements if p.phase]
        for tab, phase in zip(st.tabs(["Laning", "Mid", "Late"]), ("laning", "mid", "late")):
            with tab:
                ps = [p for p in phased if p.phase == phase]
                markers = [(p.percentile, _STANDING_COLOR[p.standing]) for p in ps]
                # Metric glyphs overlaid under each marker line (positioned by the marker's
                # percentile-x), so the bell itself says which curve line is which metric.
                icons = "".join(
                    f"<img class='bell-ic' src='{_metric_icon_src(p.metric)}' "
                    f"style='left:{_bell_x_frac(p.percentile) * 100:.1f}%'/>"
                    for p in ps
                )
                cards = "".join(
                    "<div class='pcard'>"
                    "<div class='pcard-body'>"
                    f"<div class='pcard-title'>{_STAT_LABELS[p.metric]}</div>"
                    f"<div class='pcard-val'>{_val(p.metric, p.value)}</div>"
                    f"<div class='pcard-cap'>peers {_val(p.metric, p.median)}</div>"
                    "</div>"
                    f"<div class='pcard-pct'>{_ord(p.percentile)}"
                    "<span class='pct-lbl'>percentile</span></div>"
                    "</div>"
                    for p in ps
                )
                st.html(f"<div class='bellrow'>"
                        f"<div class='bell-wrap'>{_bell_svg(markers, vb_w=560, vb_h=96)}{icons}{legend}</div>"
                        f"<div class='pcards'>{cards}</div></div>")

        st.html("<div class='perf-gap'></div>")  # flex spacer: slack between tiers 1 and 2

        # Tier 2 -- Full game vs. Peers: bell-in-card (same bell, smaller + static,
        # single shaded marker) for the non-phased cohort KPIs.
        st.html("<div class='tier-h'>Full game vs. Peers</div>")
        overall = [p for p in placements if p.phase is None]
        for col, p in zip(st.columns(len(overall)), overall):
            with col:
                st.html(
                    "<div class='bwrap'>"
                    # Bell ABOVE the card (a sibling, not inside it).
                    + _bell_svg([(p.percentile, _STANDING_COLOR[p.standing])],
                                shade_pct=p.percentile, vb_w=220, vb_h=94)
                    + "<div class='bcard'>"
                    f"<img class='card-ic' src='{_metric_icon_src(p.metric)}'/>"
                    "<div class='bcard-body'>"
                    f"<div class='bcard-title'>{_STAT_LABELS[p.metric]}</div>"
                    "<div class='bcard-valrow'>"
                    "<div class='bcard-valcol'>"
                    f"<div class='bcard-val'>{_val(p.metric, p.value)}</div>"
                    f"<div class='bcard-cap'>peers {_val(p.metric, p.median)}</div>"
                    "</div>"
                    f"<div class='bcard-pct'>{_ord(p.percentile)}"
                    "<span class='pct-lbl'>percentile</span></div>"
                    "</div>"
                    "</div></div></div>"
                )

        st.html("<div class='perf-gap'></div>")  # flex spacer: slack between tiers 2 and 3

        # Tier 3 -- Other stats: plain cards, no bell (champion-confounded, so there
        # is nothing honest to compare against a cohort).
        st.html("<div class='tier-h'>Other stats</div>")
        cards = "".join(
            "<div class='scard'>"
            f"<img class='card-ic' src='{_metric_icon_src(_DESC_METRIC[r['metric']])}'/>"
            "<div class='scard-body'>"
            f"<div class='scard-lbl'>{_DESC_DISPLAY.get(r['metric'], r['metric'])}</div>"
            f"<div class='scard-val'>{r['you']}</div></div></div>"
            for r in descriptive
            if r["metric"] not in _HIDDEN_DESCRIPTIVE  # hidden from the UI; still in the LLM context
        )
        st.html(f"<div class='scards'>{cards}</div>")


# --- coach chat --------------------------------------------------------------

def _feedback(idx: int, msg: dict, conn) -> None:
    """Thumbs rating on one answer, logged to the P0-8 feedback table when the DB is up.
    Stable key per message so the rating survives reruns and is logged once."""
    val = st.feedback("thumbs", key=f"fb_{idx}")
    logged = f"logged_{idx}"
    if val is not None and not st.session_state.get(logged):
        rating = 1 if val == 1 else -1
        if conn is not None and msg.get("conv_id"):
            try:
                db.save_feedback(conn, msg["conv_id"], rating)
                st.toast("Thanks — feedback logged.")
            except Exception:
                st.toast("Feedback could not be saved.")
        else:
            st.toast("Thanks! (monitoring off — run via docker compose to record)")
        st.session_state[logged] = True


def _highlights(subject, cohort):
    """The subject's strongest and weakest cohort placements (by percentile), each as
    a (label, ordinal) pair -- fills the coach's empty resting state with real data."""
    ps = [p for p in place_metrics(subject, cohort) if p.percentile is not None]
    if not ps:
        return None
    label = lambda p: _STAT_LABELS.get(p.metric, p.metric)
    strong = max(ps, key=lambda p: p.percentile)
    weak = min(ps, key=lambda p: p.percentile)
    return (label(strong), _ord(strong.percentile)), (label(weak), _ord(weak.percentile))


def _coach_chat(block: str, conn, highlights=None) -> None:
    """The grounded RAG Q&A, scoped to the right column. History lives in session
    state; a new question renders a compact 'Coaching…' status while the (blocking)
    answer call runs, then the cited answer + a thumbs rating."""
    msgs = st.session_state.setdefault("chat_msgs", [])

    with st.container(border=True, height="stretch", key="coach"):
        with st.container(key="coachlogo", horizontal=True,
                          horizontal_alignment="center"):
            st.image(str(ROOT / "assets" / "carryia-logo.png"))

        box = st.container(height="stretch", key="chatlog")
        with box:
            for i, m in enumerate(msgs):
                with st.chat_message(m["role"], avatar=_AVATARS.get(m["role"])):
                    st.markdown(m["content"])
                    if m["role"] == "assistant":
                        _feedback(i, m, conn)

            # An unanswered trailing user turn -> generate now (live status).
            if msgs and msgs[-1]["role"] == "user":
                with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
                    with st.status(":shimmer[Coaching…]", type="compact") as status:
                        try:
                            coach = _coach(conn)
                            # earlier turns give the coach chat memory (prepended to the
                            # answer prompt); coaching advice still grounds solely on the
                            # retrieved, cited tips.
                            answer = coach.rag(msgs[-1]["content"], personal_context=block,
                                               history=msgs[:-1])
                            conv_id = getattr(coach, "last_conversation_id", None)
                            status.update(label="Coached", state="complete")
                        except Exception as exc:
                            status.update(label="Couldn't reach the LLM", state="error")
                            st.error(
                                f"The coach could not reach the LLM (backend: `{backend()}`). "
                                "Check your credentials in `.env` (copy `.env.example`, set the "
                                "key for your backend), then ask again."
                            )
                            st.caption(f"Details: {exc}")
                            answer = None
                    if answer is not None:
                        msgs.append({"role": "assistant", "content": answer, "conv_id": conv_id})
                        st.rerun()

            # Empty-state suggestions: a vertical stack. Rendered INSIDE the chatlog (an
            # empty container renders no DOM node, so the stretch box would collapse and the
            # cluster would rise to the top) and pushed to the box's bottom via margin-top:auto
            # (CSS), so they sit just above the input. Vanish after the first message.
            if not msgs:
                # Resting state: a short intro + two real data points, so the screen never
                # reads as an empty void before the first question.
                hl = ""
                if highlights:
                    (s_lbl, s_pct), (w_lbl, w_pct) = highlights
                    hl = (
                        "<div class='intro-hl'>"
                        f"<div class='intro-chip up'><span>Strongest</span>{s_lbl} · {s_pct}</div>"
                        f"<div class='intro-chip down'><span>Focus here</span>{w_lbl} · {w_pct}</div>"
                        "</div>"
                    )
                st.html(
                    "<div class='intro'>"
                    "<div class='intro-lead'>Your post-game coach</div>"
                    "<div class='intro-sub'>Grounded in your recent games and a curated coaching "
                    "corpus — ask anything about your play.</div>"
                    f"{hl}</div>"
                )
                with st.container(key="suggestions"):
                    for label, question in SUGGESTIONS.items():
                        if st.button(label, key=f"sug_{label}", width="content"):
                            msgs.append({"role": "user", "content": question})
                            st.rerun()

        if prompt := st.chat_input("Ask Carryia...", submit_mode="disable"):
            msgs.append({"role": "user", "content": prompt})
            st.rerun()


# User avatar is the subject's own Data Dragon profile icon (hotlinked URL, from the
# committed profile.json); assistant is the committed Carryia mark. avatar= takes anything
# st.image supports -- a URL or a local path. _profile() is cached, safe to call here.
_AVATARS = {
    "user": profile_icon(_profile()["profile_icon_id"]),
    "assistant": str(ROOT / "assets" / "assistant-icon.png"),
}

# Decorative top-bar glyphs (base64 data URIs, so the raw svg never reaches the st.html
# sanitizer -- same pattern as the metric glyphs / bell). Magnifier for the fake search box:
_SEARCH_ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAyNCAyNCcgZmlsbD0nbm9uZScgc3Ryb2tlPScjOWE4YTYzJyBzdHJva2Utd2lkdGg9JzInIHN0cm9rZS1saW5lY2FwPSdyb3VuZCcgc3Ryb2tlLWxpbmVqb2luPSdyb3VuZCc+PGNpcmNsZSBjeD0nMTEnIGN5PScxMScgcj0nNycvPjxsaW5lIHgxPScyMScgeTE9JzIxJyB4Mj0nMTYuNjUnIHkyPScxNi42NScvPjwvc3ZnPg=="
# Amber caution triangle for the "Demo mode" badge:
_DEMO_ICON = ("data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIH"
              "ZpZXdCb3g9JzAgMCAyNCAyNCcgZmlsbD0nbm9uZScgc3Ryb2tlPScjZTBhODNkJyBzdHJva2Utd2lkdGg9"
              "JzInIHN0cm9rZS1saW5lY2FwPSdyb3VuZCcgc3Ryb2tlLWxpbmVqb2luPSdyb3VuZCc+PHBhdGggZD0nTT"
              "EwLjI5IDMuODYgMS44MiAxOGEyIDIgMCAwIDAgMS43MSAzaDE2Ljk0YTIgMiAwIDAgMCAxLjcxLTNMMTMu"
              "NzEgMy44NmEyIDIgMCAwIDAtMy40MiAweicvPjxsaW5lIHgxPScxMicgeTE9JzknIHgyPScxMicgeTI9Jz"
              "EzJy8+PGxpbmUgeDE9JzEyJyB5MT0nMTcnIHgyPScxMi4wMScgeTI9JzE3Jy8+PC9zdmc+")


# --- page --------------------------------------------------------------------

st.set_page_config(page_title="Carryia — Profile", page_icon="🎮", layout="wide")
_css()

games, subject, cohort, judged, descriptive, meta, block = _load()
name, _, tag = meta["riot_id"].partition("#")
profile = _profile()
conn = _db_conn()

# Top bar: a fake profile-search box + a "Demo mode" caution badge, side by side -- decorative
# only ("UI completion"). ONE st.html flex block so the two sit beside each other reliably (a
# Streamlit horizontal container did not). Row height is reserved in --searchh so the
# viewport-fit layout still fits 100vh. See _css for .topbar/.searchbox/.demo-badge.
st.html(
    "<div class='topbar'>"
    f"<div class='searchbox'><img class='search-ic' src='{_SEARCH_ICON}'/>"
    "<span class='search-ph'>Game name #Tagline</span></div>"
    f"<div class='demo-badge'><img class='demo-ic' src='{_DEMO_ICON}'/>"
    "<span>Demo mode</span></div>"
    "</div>"
)

main, chat = st.columns([0.76, 0.24], gap="medium")
with main:
    _header(name, tag, meta["region"], profile)
    st.write("")
    rail, body = st.columns([0.34, 0.66], gap="medium")
    with rail:
        _recent_games(games)
    with body:
        _performance(subject, cohort, descriptive)
with chat:
    _coach_chat(block, conn, _highlights(subject, cohort))
