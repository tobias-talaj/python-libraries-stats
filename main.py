import argparse

from utils import setup_logger, find_python_files, load_library_reference
from lib_elements_counter import process_files_in_parallel, process_file_full_analysis, process_file_simple_analysis, concatenate_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library_pickle_path", default="./api_reference_pickles/standard_library.pickle", help="Path to the pickle file containing API reference")
    parser.add_argument("--output_parquet_path", default="./data/py_imports_python_repos.parquet", help="Path and/or the filename for the output")
    parser.add_argument("--input_python_files_path", default="/media/tobiasz/crucial/python_repos/", help="Path to analysed repositories")
    parser.add_argument("--mode", default="imports", choices=["full", "simple"], help="Mode of operation: 'full' for full analysis or 'imports' for filenames and imports only")
    args = parser.parse_args()

    logger = setup_logger()

    print("Loading library reference...")
    lib_dict = load_library_reference(args.library_pickle_path)
    print("Updating list of Python files...")
    code_files = find_python_files(args.input_python_files_path, filetype='.py')

    if args.mode == "full":
        print("Counting library components occurrences...")
        df_list = process_files_in_parallel(process_file_full_analysis, lib_dict, code_files, logger)
    elif args.mode == "simple":
        print("Extracting import information...")
        df_list = process_files_in_parallel(process_file_simple_analysis, lib_dict, code_files, logger)
    

    print("Saving data to parquet...")
    concatenate_and_save(df_list, args.output_parquet_path)
    print("DONE")

if __name__ == "__main__":
    main()