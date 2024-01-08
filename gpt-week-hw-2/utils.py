import json


def load_jsonl_with_indent(file_path):
    results = []
    with open(file_path, 'r') as fin:
        lines = []
        for line in fin.readlines():
            lines.append(line)
            if line.strip('\n') == '}':
                results.append(json.loads(''.join(lines)))
                lines = []
    return results


def load_jsonl(file_path):
    results = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            results.append(json.loads(line))
    return results


def dump_jsonl(data, file_path):
    with open(file_path, 'w') as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + '\n')