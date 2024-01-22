from ctapipe.core import Tool, traits

from ctapipe.core.traits import List, Unicode


from argparse import ArgumentParser
from pathlib import Path

import sys
import gzip
import pickle
import warnings


class TemplateMerger(Tool):

    """
    Tool to merge multiple templates dictionaries each with a different
    subset of keys into one large template dictionary.
    This is useful mostly because creating multiple subsets of the template dictionary
    in parallel is much faster computationally than directly creating one large dictionary.
    """

    input_dir = traits.Path(
        default_value=None,
        help="Input directory conatining input files",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input template dictionary files",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.template.gz",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    output_file = Unicode(default_value=".", help="base output file name").tag(
        config=True
    )

    def setup(self):

        """Sets up the tool and parses input files."""

        # Load in all input files
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)

    def start(self):
        """Main rout ine of the merge tool, merges the templates"""
        self.full_template_dict = {}
        # Open and unpickle the tempalte ditionaries
        for input_file in self.input_files:
            this_dict = None
            with gzip.open(input_file, "r") as gz_inp_file:
                this_dict = pickle.load(gz_inp_file)

            for key, value in this_dict.items():
                # We need to make sure that we have a unique template for every key.
                # It is at this stage not obvious how to combine two templates for the same key into one.
                # Therefore, we throw a warning as soon as a key that already exists is accessed again.
                if key not in self.full_template_dict.keys():
                    # Write the dictionary to the common dictionary.
                    self.full_template_dict[key] = value
                else:
                    warnings.warn(
                        "The key {} is already in the merged template dictionary.".format(
                            key
                        )
                    )

    def finish(self):
        """Finish the merge tool, write template dictionary to file"""
        # Save the combined template
        file_handler = gzip.open(self.output_file + ".template.gz", "wb")
        pickle.dump(self.full_template_dict, file_handler)
        file_handler.close()


def main():
    """run the tool"""

    tool = TemplateMerger()
    tool.run()


if __name__ == "__main__":
    main()
