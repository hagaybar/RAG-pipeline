import setuptools
import os

def get_requirements():
    """Reads requirements.txt and returns a list of dependencies."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return requirements
    except FileNotFoundError:
        print("Warning: requirements.txt not found. Proceeding with an empty dependency list.")
        return []

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rag_email_pipeline",
    version="0.1.0",
    author="[Placeholder Name/Team]", # Replace with actual author or team name
    author_email="[placeholder_email@example.com]", # Replace with actual email
    description="A RAG pipeline for processing and querying email data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[placeholder_url_to_project_repo]", # Replace with actual project URL
    packages=setuptools.find_packages(where="scripts"),
    package_dir={"": "scripts"},
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha", # Or other appropriate status
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", # Or specify if OS-dependent (e.g., "Operating System :: Microsoft :: Windows")
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10",
    project_urls={ # Optional
        "Bug Tracker": "[placeholder_bug_tracker_url]",
        "Documentation": "[placeholder_documentation_url]",
        "Source Code": "[placeholder_url_to_project_repo]",
    },
)
