#!/usr/bin/env python3
"""
Test script to verify EvolveBench setup
"""

import json
import yaml
import os
from pathlib import Path


def test_templates():
    """Test that all templates exist and are valid YAML."""
    templates_dir = Path(__file__).parent.parent / "templates"
    template_files = [
        "config_handwritten.yaml",
        "config_llm_generated.yaml",
        "config_llm_judge.yaml",
    ]

    print("🔍 Testing templates...")
    for template_file in template_files:
        template_path = templates_dir / template_file
        if not template_path.exists():
            print(f"❌ Template not found: {template_path}")
            return False

        try:
            with open(template_path, "r") as f:
                yaml.safe_load(f)
            print(f"✅ {template_file} is valid YAML")
        except yaml.YAMLError as e:
            print(f"❌ {template_file} has YAML error: {e}")
            return False

    return True


def test_tasks():
    """Test that all tasks have required files."""
    tasks_dir = Path(__file__).parent.parent / "tasks"
    required_files = [
        "task.yaml",
        "config_handwritten.yaml",
        "config_llm_generated.yaml",
        "config_llm_judge.yaml",
    ]

    print("\n🔍 Testing tasks...")
    task_dirs = [
        d for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    for task_dir in task_dirs:
        print(f"  Testing {task_dir.name}...")

        # Check required files
        for required_file in required_files:
            file_path = task_dir / required_file
            if not file_path.exists():
                print(f"    ❌ Missing: {required_file}")
                return False
            print(f"    ✅ {required_file}")

        # Validate task.yaml
        try:
            with open(task_dir / "task.yaml", "r") as f:
                task_meta = yaml.safe_load(f)

            required_fields = [
                "task_id",
                "description",
                "evaluation_criteria",
                "task_config",
            ]
            for field in required_fields:
                if field not in task_meta:
                    print(f"    ❌ Missing field in task.yaml: {field}")
                    return False

            print("    ✅ task.yaml is valid")

        except yaml.YAMLError as e:
            print(f"    ❌ task.yaml has YAML error: {e}")
            return False

    return True


def test_registry():
    """Test that registry.json is valid."""
    print("\n🔍 Testing registry...")

    registry_path = Path(__file__).parent.parent / "registry.json"
    if not registry_path.exists():
        print("❌ registry.json not found")
        return False

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)

        if not isinstance(registry, list) or len(registry) == 0:
            print("❌ registry.json should be a non-empty list")
            return False

        # Check required fields
        required_fields = [
            "name",
            "version",
            "description",
            "task_ids",
            "evaluator_approaches",
        ]
        for field in required_fields:
            if field not in registry[0]:
                print(f"❌ Missing field in registry: {field}")
                return False

        print("✅ registry.json is valid")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ registry.json has JSON error: {e}")
        return False


def test_package():
    """Test that package structure is correct."""
    print("\n🔍 Testing package structure...")

    base_path = Path(__file__).parent.parent
    package_files = [
        "evolve_bench/__init__.py",
        "evolve_bench/cli.py",
        "evolve_bench/harness.py",
        "pyproject.toml",
        "README.md",
    ]

    for file_path in package_files:
        if not (base_path / file_path).exists():
            print(f"❌ Missing package file: {file_path}")
            return False
        print(f"✅ {file_path}")

    return True


def test_api_key():
    """Test that OPENAI_API_KEY is set for LLM-based runs."""
    print("\n🔍 Testing API credentials...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "❌ OPENAI_API_KEY is not set. LLM-based evaluators will fail (401/timeout)."
        )
        return False
    print("✅ OPENAI_API_KEY is set")
    return True


def main():
    """Run all tests."""
    print("🧪 EvolveBench Setup Test")
    print("=" * 30)

    tests = [
        test_templates,
        test_tasks,
        test_registry,
        test_package,
        test_api_key,
    ]

    all_passed = True
    for test in tests:
        if not test():
            all_passed = False

    print("\n" + "=" * 30)
    if all_passed:
        print("🎉 All tests passed! EvolveBench is ready to use.")
    else:
        print("❌ Some tests failed. Please check the setup.")

    return all_passed


if __name__ == "__main__":
    main()
