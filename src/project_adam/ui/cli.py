import torch
from ..config import get_memory_dir


def run_cli(agent):
    greeting = agent.persona.get_opening() if agent.persona else ""
    print("\n── Adam (COGNET) ──")
    if greeting:
        print(f"Adam: {greeting}")
    print("Commands: /search <q> /memory /schemas /persona /users /profile /stats /save /exit\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            agent.episodic_memory.save()
            agent.semantic_memory.save()
            agent.user_profiles.save()
            agent._save_adapter()
            closing = agent.persona.get_closing() if agent.persona else "Goodbye."
            print(f"Adam: One more thing — {closing}")
            break
        if user.startswith("/search "):
            q = user[8:]
            print(f"  [web] searching: {q}")
            r = agent.web_search.search(q)
            print(f"  {r[:400] if r else 'no results'}")
            if r:
                agent.episodic_memory.add(f"[web: {q}] {r[:200]}", reward=0.5)
            continue
        if user == "/memory":
            print(f"  Episodic: {len(agent.episodic_memory.episodes)} items")
            for ep in agent.episodic_memory.episodes[-5:]:
                r = ep.get("reward", 0)
                t = ep.get("text", "")[:80]
                print(f"    [r={r:.1f}] {t}")
            continue
        if user == "/schemas":
            print(f"  Semantic: {len(agent.semantic_memory.schemas)} schemas")
            for cat, data in agent.semantic_memory.schemas.items():
                print(f"    {cat}: {'; '.join(data['facts'][-3:])}")
            continue
        if user == "/persona":
            if agent.persona:
                print(f"  Persona: {agent.persona.path.name}")
                print(f"  Essence: {agent.persona.essence[:100]}")
                print(f"  Behavior rules: {len(agent.persona.behavior_rules)}")
                sig_count = len(agent.persona.language_patterns.split()) if agent.persona.language_patterns else 0
                print(f"  Signature expressions: ~{sig_count}")
            else:
                print("  No persona loaded")
            continue
        if user == "/stats":
            s = agent.metacognitive.stats()
            q = agent.current_profile.get("last_q", "N/A") if agent.current_profile else "N/A"
            print(f"  Interactions: {s['total']} | Avg confidence: {s['avg_confidence']} | Slow path: {s['slow_path_rate']}")
            print(f"  Last SFL Q: {q} | Custom rules: {len(agent.current_profile.get('custom_rules', [])) if agent.current_profile else 0}")
            continue
        if user == "/dashboard":
            p = agent.current_profile or {}
            s = agent.metacognitive.stats()
            print("  ┌─ DASHBOARD ──────────────────────────────")
            print(f"  │ User: {p.get('name', '—')}")
            print(f"  │ Interactions: {p.get('interaction_count', 0)}")
            print(f"  │ Avg sentiment: {p.get('avg_sentiment', 0):.2f}")
            print(f"  │ Avg confidence: {s.get('avg_confidence', 0)}")
            print(f"  │ Last action: {s.get('last_action', '—')}")
            print(f"  │ Slow path: {s.get('slow_path_rate', 0)}")
            q_hist = p.get('q_history', [])
            r_hist = p.get('reward_history', [])
            if q_hist:
                print(f"  │ SFL Q (last 20): {'█' * int(abs(q_hist[-1]) * 10)} {q_hist[-1]:.2f}")
            if r_hist:
                recent = r_hist[-20:]
                avg_r = sum(recent) / len(recent)
                print(f"  │ Reward trend: ↑{sum(1 for r in recent if r > 0)} ↓{sum(1 for r in recent if r < 0)} "
                      f"avg={avg_r:.2f} last={r_hist[-1]:.2f}")
            c_hist = s.get('confidence_history', [])
            if c_hist:
                print(f"  │ Confidence: avg={sum(c_hist)/len(c_hist):.2f} "
                      f"last={c_hist[-1]:.2f} low={sum(1 for c in c_hist if c < 0.3)}")
            rw = p.get('rule_weights', {})
            if rw:
                vals = list(rw.values())
                print(f"  │ Rule weights: max={max(vals):.2f} min={min(vals):.2f} "
                      f"spread={max(vals)-min(vals):.2f}")
            print("  └──────────────────────────────────────────")
            continue
        if user == "/save":
            agent.episodic_memory.save()
            agent.semantic_memory.save()
            agent.user_profiles.save()
            print("  [saved all memory systems]")
            continue
        if user == "/users":
            users = agent.user_profiles.list_users()
            print(f"  Known users ({len(users)}):")
            for u in users:
                p = agent.user_profiles.get_or_create(u)
                print(f"    {u}: {p.get('interaction_count', 0)} interactions, sentiment={p.get('avg_sentiment', 0):.2f}")
            continue
        if user == "/profile":
            p = agent.user_profiles.get_current()
            if p:
                print(f"  Current user: {p['name']}")
                print(f"  Interactions: {p['interaction_count']}")
                print(f"  Avg sentiment: {p['avg_sentiment']:.2f}")
                print(f"  Last SFL Q-value: {p.get('last_q', 'N/A')}")
                print(f"  Custom rules: {len(p.get('custom_rules', []))}")
                top_topics = sorted(p['topics'].items(), key=lambda x: -x[1])[:5]
                if top_topics:
                    print(f"  Top topics: {', '.join(f'{t}({c})' for t, c in top_topics)}")
                adopted = [ph for ph, d in p['adopted_phrases'].items() if d.get('count', 0) >= 5]
                if adopted:
                    print(f"  Adopted phrases (\u22655 uses): {len(adopted)}")
                rw = p.get('rule_weights', {})
                if rw:
                    top_rule = max(rw.items(), key=lambda x: x[1])
                    bot_rule = min(rw.items(), key=lambda x: x[1])
                    print(f"  Strongest rule: rule {top_rule[0]} (w={top_rule[1]:.2f})")
                    print(f"  Weakest rule: rule {bot_rule[0]} (w={bot_rule[1]:.2f})")
            else:
                print("  No current user")
            continue

        agent.chat(user)
