"""The solutions notebook is canonical. Only exercise source and outputs change."""
from pathlib import Path
import copy
import nbformat

ROOT = Path(__file__).resolve().parent.parent

def sync_student():
    solution = nbformat.read(ROOT / 'seminar5_workbook_solutions.ipynb', as_version=4)
    student = copy.deepcopy(solution)
    for cell in student.cells:
        if cell.cell_type == 'code':
            if 'student_source' in cell.metadata:
                cell.source = cell.metadata.student_source
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(student)
    nbformat.write(student, ROOT / 'seminar5_workbook.ipynb')
    print('Student notebook synced from current solutions; prose preserved.')

if __name__ == '__main__':
    sync_student()
