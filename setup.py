from setuptools import setup

setup(
    name="template_builder",
    version="2.0",
    packages=["template_builder"],
    package_dir={"template_builder": "template_builder"},
    package_data={
        "template_builder": [
            "configs/array_trigger_temp.dat",
            "configs/cta-temp_run.sh",
            "configs/run_sim_template",
            "configs/simtel_template.cfg",
        ],
        "template_builder_data": ["data/gamma_HESS_example.simhess.gz"],
    },
    include_package_data=True,
    url="",
    license="",
    author="parsonsrd",
    author_email="",
    description="Creation tools for building ImPACT templates for ctapipe",
    entry_points={
        "console_scripts": [
            "template-fitter = template_builder.template_fitter:main",
            "template-merger = template_builder.merge_templates:main",
        ]
    },
)
