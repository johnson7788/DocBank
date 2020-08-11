## DocBank dataset samples


我们使用原始的pdf名称作为前缀，并且页面索引从0开始。

每个样本都包含六个文件：

- {prefix}_color.pdf: 将结构的字体颜色更改为特定于结构的颜色后生成的pdf文件。
- {prefix}_black.pdf: 将结构的字体颜色更改为黑色后生成的pdf文件，与原始pdf相似。
- {prefix}_{page_index}.jpg: “ _ color.pdf”中页面的图像。
- {prefix}_{page_index}_ori.jpg: “ _ black.pdf”中页面的图像。
- {prefix}_{page_index}_ann.jpg: 此示例页面的注释图。

出于可视化目的给出了前五个文件，以下文件是模型所需的唯一文件。
- {prefix}_{page_index}.txt: 此示例页面的注释。

### Annotation Format

每行包含一个token及其以下信息：
- bounding box ((x0, y0), (x1, y1)) - > (x0, y0, x1, y1)
- color (R, G, B)
- font
- label

| Index   | 0     | 1  | 2  | 3  | 4  | 5 | 6 | 7 | 8         | 9     |
|---------|-------|----|----|----|----|---|---|---|-----------|-------|
| Content | token | x0 | y0 | x1 | y1 | R | G | B | font name | label |