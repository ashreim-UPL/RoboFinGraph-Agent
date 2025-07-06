export default function cleanMermaid(str) {
  str = str.replace(/^---[\s\S]+?---/, '');
  str = str.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
  return str.trim();
}
