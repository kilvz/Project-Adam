def run_web_ui(agent):
    import gradio as gr

    def respond_stream(message, history):
        full_reply = ""
        def on_token(tok):
            nonlocal full_reply
            full_reply += tok
        agent.chat(message, token_callback=on_token)
        yield full_reply

    def get_dashboard():
        sfl = agent.sfl_module
        rows = []
        if sfl.q_history:
            rows.append(f"**SFL Q** (last 20): {', '.join(f'{q:.2f}' for q in sfl.q_history[-20:])}")
            if len(sfl.q_history) >= 10:
                rows.append(f"Q trend: {'↑' if sfl.q_history[-1] > sfl.q_history[-10] else '↓'}"
                            f" ({sfl.q_history[-10]:.2f} → {sfl.q_history[-1]:.2f})")
        if sfl.confidence_history:
            c = sfl.confidence_history
            rows.append(f"**Confidence**: avg={sum(c)/len(c):.2f} last={c[-1]:.2f}")
        profile = agent.current_profile
        if profile:
            rw = profile.get("rule_weights", {})
            if rw:
                vals = list(rw.values())
                rows.append(f"**Rules**: {len(vals)} active, spread={max(vals)-min(vals):.2f}")
            topics = profile.get("topics", {})
            if topics:
                top = sorted(topics.items(), key=lambda x: -x[1])[:5]
                rows.append(f"**Topics**: {', '.join(f'{t}({c})' for t,c in top)}")
            rows.append(f"**Interactions**: {profile.get('interaction_count', 0)}")
            rows.append(f"**Avg sentiment**: {profile.get('avg_sentiment', 0):.2f}")
            custom = profile.get("custom_rules", [])
            if custom:
                rows.append(f"**Custom rules**: {len(custom)}")
            adopted = [p for p, d in profile.get("adopted_phrases", {}).items()
                       if d.get("count", 0) >= 5]
            if adopted:
                rows.append(f"**Adopted phrases**: {len(adopted)}")
        if agent.current_profile:
            rows.append(f"**User**: {agent.current_profile.get('name', 'unknown')}")
        if agent.action_selector and hasattr(agent.action_selector, '_last_meta_action'):
            rows.append(f"**Last action**: {agent.action_selector._last_meta_action}")
        return "\n\n".join(rows) if rows else "No data yet."

    def get_memory():
        parts = []
        epi = agent.episodic_memory
        if epi.episodes:
            parts.append(f"**Episodic**: {len(epi.episodes)} episodes")
            for e in epi.episodes[-3:]:
                txt = e.get("text", "")[:60]
                rwd = e.get("reward", 0)
                parts.append(f"- `{txt}...` (r={rwd:.2f})")
        sem = agent.semantic_memory
        if sem.schemas:
            parts.append(f"**Semantic**: {len(sem.schemas)} schemas")
            for cat, data in list(sem.schemas.items())[:5]:
                facts = "; ".join(data.get("facts", [])[-2:])[:80]
                if facts:
                    parts.append(f"- **{cat}**: {facts}")
        users = agent.user_profiles
        if users and hasattr(users, 'profiles'):
            parts.append(f"**Users**: {len(users.profiles)} profiles")
            for name, p in list(users.profiles.items())[:5]:
                parts.append(f"- **{name}**: {p.get('interaction_count', 0)} turns, "
                             f"sentiment={p.get('avg_sentiment', 0):.2f}")
        return "\n\n".join(parts) if parts else "Empty."

    with gr.Blocks(title="Project Adam", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Project Adam — COGNET")

        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=respond_stream, type="tuples",
                title="Adam", description="Conversational AI that adapts to you.",
            )

        with gr.Tab("Dashboard"):
            dash_btn = gr.Button("Refresh")
            dash_out = gr.Markdown("Click Refresh to load.")
            dash_btn.click(fn=get_dashboard, outputs=dash_out)
            demo.load(fn=get_dashboard, outputs=dash_out)

        with gr.Tab("Memory"):
            mem_btn = gr.Button("Refresh")
            mem_out = gr.Markdown("Click Refresh to load.")
            mem_btn.click(fn=get_memory, outputs=mem_out)
            demo.load(fn=get_memory, outputs=mem_out)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
