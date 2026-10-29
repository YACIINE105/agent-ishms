import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from controllers.ProcessController import ProcessController
from langchain_core.documents import Document

controller = ProcessController.__new__(ProcessController)

pdf_pages = [
    Document(page_content='Page 1 text. ' * 50, metadata={'page': 1, 'tables': [{'table_id': 'p1-t1'}]}),
    Document(page_content='Page 2 text. ' * 50, metadata={'page': 2, 'tables': []})
]
pdf_chunks = controller.process_file_content(pdf_pages, 'sample.pdf', chunk_size=150, overlap_size=20)
print(f'PDF Chunks: {len(pdf_chunks)}')
assert pdf_chunks[0].metadata['page'] == 1
assert 'tables' in pdf_chunks[0].metadata
print('First PDF chunk metadata:', pdf_chunks[0].metadata)
print('✓ PDF metadata preserved across chunks!')

csv_rows = [
    Document(page_content='ID: 101, Service: Card, Fee: 500 DZD', metadata={'row': 0}),
    Document(page_content='ID: 102, Service: Cert, Fee: 200 DZD', metadata={'row': 1})
]
csv_chunks = controller.process_file_content(csv_rows, 'sample.csv', chunk_size=200)
print(f'CSV Chunks: {len(csv_chunks)}')
assert csv_chunks[0].metadata['row'] == 1
assert csv_chunks[1].metadata['row'] == 2
print('CSV Chunk 0 metadata:', csv_chunks[0].metadata)
print('CSV Chunk 1 metadata:', csv_chunks[1].metadata)
print('✓ CSV row boundaries & 1-based indices preserved!')
print('\nAll tests passed successfully!')
