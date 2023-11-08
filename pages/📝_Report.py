import os
import re
import base64
import streamlit as st


def markdown_images(markdown):
    images = re.findall(
        r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )
    return images


def img_to_bytes(img_path):
    with open(f"{img_path}", "rb") as image:
        img_bytes = image.read()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<div style="text-align: center;"><img src="data:{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 85%;"></div><div style="text-align: left;"></div><br>'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = f"./pages/{image[2]}"
        if os.path.exists(image_path):
            markdown = markdown.replace(
                image_markdown, img_to_html(image_path, image_alt)
            )
    return markdown


def main() -> None:
    ####################
    # Config
    ####################
    st.set_page_config(
        page_title="Analysis",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    with open("./pages/report.md", "r", encoding="utf8") as file:
        page = file.read()

    page = markdown_insert_images(page)

    st.markdown(page, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
