from pdf2image import convert_from_path

# 将 FedGUI 论文转换为图片列表
images = convert_from_path('fedgui.pdf')

# 保存第一页（通常是项目架构图所在的页面）
images[0].save('fedgui_architecture.png', 'PNG')

# 批量保存所有页面
for i, image in enumerate(images):
    image.save(f'paper_page_{i+1}.png', 'PNG')