# data/ 目录说明

把你自己的文档（PDF / DOCX / MD / TXT）放到本目录下任意位置，运行
`python app.py ingest` 即可索引入库。
`chroma/`：运行后自动生成的向量库持久化目录，不要手动编辑。

重新放入新文档后，再次执行 `python app.py ingest` 即可更新索引。
