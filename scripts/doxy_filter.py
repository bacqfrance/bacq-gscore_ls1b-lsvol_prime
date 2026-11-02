#!/usr/bin/env python3
import sys
import re

def process_python(content):
	# Match une fonction def avec son éventuel bloc ## d'en-tête et sa docstring
	pattern_function = re.compile(
		r'^(?P<header_doc>(?:[ \t]*##[^\n]*\n)+)?'                             # Lines ## au-dessus
		r'(?P<indent>[ \t]*)def\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\)\s*(?:->[^:]+)?:)' # Capture la signature
		r'(?P<after_def>[^\n]*\n)'                                             # Fin de ligne def
		r'(?P<docstring>\s*(?:r?"""|r?\'\'\')[\s\S]*?(?:"""|\'\'\')\s*\n)?',  # Docstring optionnelle
		re.MULTILINE
	)
	
	if not re.search(r'^\s*def\s+', content, re.MULTILINE):
		return content
	
	output = []

	# Détection et conversion systématique de la docstring de module r""" (@file / @package)
	header_file_match = re.match(r'^\s*(r?"""[\s\S]*?"""|r?\'\'\'[\s\S]*?\'\'\')\s*\n', content)
	if header_file_match:
		header_text = header_file_match.group(0)
		if '@file' in header_text or '@package' in header_text:
			# Extraction du contenu textuel de la docstring
			clean_doc = re.sub(r'^\s*r?("""|\'\'\')|\s*("""|\'\'\')\s*$', '', header_text.strip())
			commented_lines = []
			for line in clean_doc.splitlines():
				stripped = line.strip()
				# Transformation en ## pour que Doxygen l'associe nativement au fichier/module
				commented_lines.append(f"## {stripped}" if stripped else "##")
				
			output.append("\n".join(commented_lines) + "\n\n")
			content = content[header_file_match.end():]
		
	matches = list(pattern_function.finditer(content))
	if not matches:
		return content
	
	last_idx = 0
	
	for i, match in enumerate(matches):
		output.append(content[last_idx:match.start()])
		
		# 1. Indentation stricte mesurée sur la ligne def
		func_indent = match.group('indent')             # Ex: "    " (4 espaces)
		body_indent = func_indent + "    "              # Ex: "        " (8 espaces)
		
		header_doc = match.group('header_doc') or ""
		existing_ds = match.group('docstring')
		func_name = match.group('name')
		func_sig = match.group('sig')
		after_def = match.group('after_def')
		
		# Portée du corps de la fonction
		next_start = matches[i+1].start() if i + 1 < len(matches) else len(content)
		body_code = content[match.end():next_start]
		
		# 2. Extraction des ## dans le corps
		body_lines = body_code.splitlines(True)
		body_docs = []
		clean_body_lines = []
		
		for line in body_lines:
			stripped = line.strip()
			if stripped.startswith("##"):
				doc_text = re.sub(r'^\s*##\s?', '', line).rstrip()
				if doc_text:
					body_docs.append(doc_text)
			else:
				clean_body_lines.append(line)
			
		clean_body = "".join(clean_body_lines)
		# Correctif 1 : Reconstruction propre sans calcul manuel d'offset
		func_decl = f"{func_indent}def {func_name}{func_sig}{after_def}"

		# Normalisation : Extraction et conversion de la docstring r""" existante en lignes ##
		ds_lines = []
		if existing_ds:
			clean_ds = re.sub(r'^\s*r?("""|\'\'\')|\s*("""|\'\'\')\s*$', '', existing_ds.strip())
			for line in clean_ds.splitlines():
				# Nettoie les tabulations/espaces de début de ligne dans la docstring
				stripped_line = line.strip()
				ds_lines.append(f"{func_indent}## {stripped_line}\n" if stripped_line else f"{func_indent}##\n")
			
		# Assemblage de l'en-tête global (Header ## existant + Docstring convertie en ##)
		base_header = header_doc if header_doc else ""
		if ds_lines:
			base_header += "".join(ds_lines)
		
		if body_docs:
			# Ajout des détails d'implémentation (body_docs) en syntaxe ##
			impl_lines = [
				f"{func_indent}##\n",
				f"{func_indent}## @par Detailed implementation:\n"
			]
			for d in body_docs:
				impl_lines.append(f"{func_indent}## {d.strip()}\n")
			
			full_header = base_header + "".join(impl_lines)
			func_block = full_header + func_decl		
		else:
 			func_block = base_header + func_decl if base_header else func_decl
		
		output.append(func_block + clean_body)
		last_idx = next_start
	
	output.append(content[last_idx:])
	return "".join(output)


def process_cpp(content):
	if "// doc:" not in content:
		return content
	
	doc_lines = re.findall(r'^\s*\/\/\s*doc:\s*(.*)$', content, flags=re.MULTILINE)
	if not doc_lines:
		return content
	
	formatted_docs = "\n * ".join(doc_lines)
	
	if re.search(r'\/\*\*[\s\S]*?\*\/', content):
		def repl_cpp(match):
			return f"{match.group(1)}\n * \n * @par Detailed implementation :\n * {formatted_docs}\n {match.group(2)}"
		return re.sub(r'(\/\*\*[\s\S]*?)(\*\/)', repl_cpp, content, count=1)
	
	return content


def main():
	if len(sys.argv) < 2:
		return
	
	filepath = sys.argv[1]
	try:
		with open(filepath, 'r', encoding='utf-8') as f:
			content = f.read()
	except Exception:
		return
	
	if filepath.endswith('.py'):
		output = process_python(content)
	elif any(filepath.endswith(ext) for ext in ['.cpp', '.hpp', '.h', '.cxx', '.cc']):
		output = process_cpp(content)
	else:
		output = content
	
	sys.stdout.write(output)


if __name__ == '__main__':
	main()
