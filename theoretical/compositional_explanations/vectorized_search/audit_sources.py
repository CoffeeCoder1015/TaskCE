"""Inspect and compile this package's source without importing its GPU code."""

import ast
from importlib.util import find_spec
from pathlib import Path


def audit_sources() -> None:
    package = Path(__file__).resolve().parent
    source_files = sorted(package.glob("*.py"))
    parsed_files = {}
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        parsed_files[path.name] = ast.parse(source, filename=str(path))
    print(f"In-memory compilation: {len(source_files)} source files passed")
    declared_functions = {
        node.name
        for tree in parsed_files.values()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for filename, tree in parsed_files.items():
        print(f"\n{filename}")
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                fields = [field.target.id for field in node.body
                          if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)]
                print(f"  type {node.name}: {', '.join(fields)}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = set()
                for call in ast.walk(node):
                    if not isinstance(call, ast.Call):
                        continue
                    target = call.func
                    if isinstance(target, ast.Subscript):
                        target = target.value
                    if isinstance(target, ast.Name) and target.id in declared_functions:
                        calls.add(target.id)
                print(f"  function {node.name} -> {', '.join(sorted(calls)) or '(no package function calls)'}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"AND", "OR", "AND_NOT", "OPERATION_COUNT"}:
                        print(f"  operation {target.id} = {ast.literal_eval(node.value)}")
    print("\nDependency discovery (without importing the search package):")
    for dependency in ("numpy", "sympy", "torch", "triton"):
        spec = find_spec(dependency)
        print(f"  {dependency}: {'discoverable' if spec is not None else 'not discoverable'}")
    print("\nScope: source inventory and syntax compilation only.")
    print("CUDA compilation, numerical behavior, and full runtime equivalence remain unverified.")


if __name__ == "__main__":
    audit_sources()
