from setuptools import setup, find_namespace_packages

setup(
    name="ortho-fastdtr",
    version="0.1.0",
    description="Ortho-FastDTR: Orthogonal Refusal Subspace & Fast Taylor Attribution for VLM Jailbreak Defense",
    packages=find_namespace_packages(include=["llava*", "DTR_util*"]),
    python_requires=">=3.8",
)
