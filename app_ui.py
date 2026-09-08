import gradio as gr
from app.pipeline.pipeline import ai_pipeline


def run_pipeline(topic: str):
    if not topic.strip():
        return "Error: Please enter a valid topic.", "", "", ""

    pipeline_state = ai_pipeline(topic)

    search_output = pipeline_state.get(
        'search_result', 'No search result found.'
    )
    scrape_output = pipeline_state.get(
        'scrape_result', 'No scrape data found.'
    )
    writer_output = pipeline_state.get(
        'writer', 'No writer output found.'
    )
    critic_output = pipeline_state.get(
        'critic', 'No critic output found.'
    )

    return search_output, scrape_output, writer_output, critic_output


# Custom CSS
css = """
.scrollable-box textarea {
    overflow-y: auto !important;
    resize: vertical !important;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css
) as demo:

    gr.Markdown("# 🤖 Multi-Agent AI Research & Writing Pipeline")

    gr.Markdown(
        "Enter any research topic below. The system will search the web, "
        "scrape content, synthesize a report using a writer agent, and "
        "critique it using a critic agent."
    )

    with gr.Row():
        topic_input = gr.Textbox(
            label="Research Topic",
            placeholder="e.g., How did the moon originate?",
            value="how did the moon originate?"
        )

    submit_btn = gr.Button("Run Pipeline", variant="primary")

    with gr.Tabs():

        with gr.TabItem("Writer Summary"):
            writer_box = gr.Textbox(
                label="Writer Output",
                lines=10,
                max_lines=25,
                interactive=False,
                elem_classes=["scrollable-box"]
            )

        with gr.TabItem("Critic Review"):
            critic_box = gr.Textbox(
                label="Critic Feedback",
                lines=10,
                max_lines=25,
                interactive=False,
                elem_classes=["scrollable-box"]
            )

        with gr.TabItem("Scraped Data"):
            scrape_box = gr.Textbox(
                label="Scraped Article Content",
                lines=10,
                max_lines=25,
                interactive=False,
                elem_classes=["scrollable-box"]
            )

        with gr.TabItem("Search Result"):
            search_box = gr.Textbox(
                label="Discovered URL",
                interactive=False,
                elem_classes=["scrollable-box"]
            )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[topic_input],
        outputs=[
            search_box,
            scrape_box,
            writer_box,
            critic_box
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
