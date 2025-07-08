import os
from PyPDF2 import PdfReader

def check_hallucination(report_dir):
    results = []
    for entry in os.scandir(report_dir):
        if entry.is_dir():
            # extract provider from folder name (suffix after last underscore)
            provider = entry.name.rsplit('_', 1)[-1]
            subdir = entry.path
            for root, _, files in os.walk(subdir):
                for fname in files:
                    if fname.lower().endswith('.pdf'):
                        path = os.path.join(root, fname)
                        try:
                            reader = PdfReader(path)
                            num_pages = len(reader.pages)
                            hallucination = num_pages > 2
                            results.append({
                                "file": path,
                                "provider": provider,
                                "pages": num_pages,
                                "hallucination": hallucination
                            })
                        except Exception as e:
                            results.append({
                                "file": path,
                                "provider": provider,
                                "error": str(e),
                                "hallucination": None
                            })
    return results

if __name__ == "__main__":
    report_dir = "./report"  # adjust path as needed
    status = check_hallucination(report_dir)
    for info in status:
        if "error" in info:
            print(f"[ERROR] ({info['provider']}) {info['file']}: {info['error']}")
        else:
            flag = "TRUE" if info["hallucination"] else "FALSE"
            print(f"{info['provider']:10s} | {info['file']} → {info['pages']} pages | hallucination = {flag}")
