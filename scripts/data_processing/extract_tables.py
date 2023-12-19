from argparse import ArgumentParser
import json
import re

from bs4 import BeautifulSoup
import pandas as pd


def soupify(table_json):
    soup = BeautifulSoup(table_json, "lxml-xml")
    for row in soup.find_all("row"):
        row.name = "tr"
    for cell in soup.find_all("cell"):
        cell.name = "td"
    for tex_math in soup.find_all("texmath"):
        # remove the "texmath"
        tex_math.extract()
    return soup


def has_x(table_soup):
    return "✗" in " ".join(table_soup.strings)


def not_too_long(table_soup):
    return len(str(table_soup)) < 5e3


def has_rows(table_soup):
    return table_soup.find("tr")


def has_max_2_sub_tables(table_soup):
    return len(table_soup.find_all("table")) <= 2


def has_at_least_2_cols(table_soup):
    # td is number of cells, so use combo of # cells and # of rows to get columns
    return len(table_soup.find_all("td")) >= 4 and len(table_soup.find_all("tr")) >= 2


def has_at_least_2_rows(table_soup):
    return len(table_soup.find_all("tr")) >= 2


def has_cites(table_soup):
    soup_text = " ".join(table_soup.strings)
    return len(table_soup.find_all("cit")) > 0 or ("et al" in soup_text)


def has_at_least_2_cites(table_soup):
    soup_text = " ".join(table_soup.strings)
    return len(table_soup.find_all("cit")) > 2


FLOAT_REGEX = re.compile("\d\.\d")


def has_no_floats(table_soup):
    for s in table_soup.strings:
        s = s.strip()
        if s and FLOAT_REGEX.search(s) is not None:
            return False
    return True


DEFAULT_TABLE_FILTERS = [
    not_too_long,
    has_at_least_2_cites,
    has_max_2_sub_tables,
    has_at_least_2_cols,
    has_at_least_2_rows,
    has_no_floats,
]


def extract_valid_tables(path, table_filters):
    """
    path by default is "arxiv_dump/out_xml/2310.00000-07773.jsonl"

    This is the output of running the following on the s2 cluster:

    1. Download the gzipped latex data into `in_latex_s3/{month}`:
    `aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/2310" in_latex_s3/2310`
    or
    `aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/2309" in_latex_s3/2309`

    The directory that's created has a bunch of .gz files in it - one with all the latex for that submission

    2. Bundle the .gz file into a tar file
    `tar cvf in_tar/2310.00000-07773.tar in_latex_s3/2310/`


    3. Extract the xml from the latex files
    `python unarXive/src/prepare.py in_tar/ out_xml/ arxiv-metadata-oai-snapshot.sqlite`

    This takes all of the latex that's packaged in `in_tar` and outputs its associated xml in `out_xml`

    Then on your local machine
    ```
    scp benjaminn@s2-cirrascale-10.reviz.ai2.in:~/nfs/arxiv_tables/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml/
    ```

    Then run this script:

    Then run `populate_bib_entries"
    Finally, run
    python scripts/data_processing/download_full_texts.py data/arxiv_tables/2308_papers.jsonl
    """
    valid_tables = []
    with open(path) as f:
        added_tables = set()
        for line in f:
            paper = json.loads(line)
            # if paper["paper_id"] != "2310.03103v1":
            #     continue
            filtered_tables = {}
            # print(len(paper["tables"]))
            for key, table in paper["tables"].items():
                if table["table"]:
                    table_soup = soupify(table["table"])

                    # Filter tables
                    exit_early = False
                    for flter in table_filters:
                        if not flter(table_soup):
                            # exit early
                            exit_early = True
                            break

                    if exit_early:
                        continue

                    # Keep the outermost table always. But prevent adding smaller tables
                    # Remove duplicates (as long as the larger table comes first, the smaller ones
                    # won't make it in). Usually the larger seems to come first.
                    if table_soup.find("table") in added_tables:
                        continue
                    else:
                        for sub_table in table_soup.find_all("table"):
                            added_tables.add(sub_table)

                    filtered_tables[key] = table
                    filtered_tables[key]["soup"] = table_soup

            if filtered_tables:
                new_paper = {k: v for k, v in paper.items() if k != "tables"}
                new_paper["tables"] = filtered_tables
                valid_tables.append(new_paper)

    return valid_tables


def soup_to_json(table_soup, verbose=False):
    # first, determine the number of columns as the max number of cells in a row
    num_cols = max(
        [len(row.find_all("td")) for row in table_soup.find_all("tr")]
        + [
            sum([int(cell.attrs.get("cols", "1")) for cell in row.find_all("td")])
            for row in table_soup.find_all("tr")
        ]
    )
    if verbose:
        print(num_cols)

    # next, extract the values. Some rows are "header" rows and contain explanatory info.
    # for now, we track these separately in a "incomplete_rows" field

    table = {"incomplete_rows": [], "table": []}
    columns = [[] for _ in range(num_cols)]
    for row_i, row in enumerate(table_soup.find_all("tr")):
        # First, we want to collapse the header rows that contain the column names
        cells = row.find_all("td")

        # skip any all-empty rows
        if all([not cell.text.strip() for cell in cells]):
            continue

        if len(cells) < num_cols:
            # incomplete rows have fewer than the max number of cells. They'll be stored
            # in a separate place or sometimes merged into the column headers
            if not (table["table"]):
                # we haven't started adding to the table, so this is most likely still a header row
                if verbose:
                    print(cells)
                last_col = 0
                for cell in cells:
                    num_spanning_cols = int(cell.attrs.get("cols", "1"))
                    for i in range(last_col, last_col + num_spanning_cols):
                        columns[i].append(cell.text)
                    last_col += num_spanning_cols

            for cell in cells:
                table["incomplete_rows"].append(
                    {
                        "row_idx": row_i,
                        "text": cell.text,
                        "cols": int(cell.attrs.get("cols", "1")),
                    }
                )
        else:
            if not table["table"]:
                # column headers
                for i, cell in enumerate(cells):
                    columns[i].append(cell.text)
                columns = ["-".join(col) for col in columns]
                table["table"].append(columns)
            else:
                table["table"].append([cell.text for cell in cells])

    # next, assume the first row has the column headers and the first col has the row headers
    if not table["table"]:
        table_dict = {}
    else:
        table_dict = pd.DataFrame(table["table"][1:], columns=table["table"][0]).to_dict(orient="list")
        # if len(table_dict) != len(table["table"][0]):
        #     table_dict = {}
    table["table_dict"] = table_dict
    return table


def create_dataset(tables_by_paper):
    """
    Flattens the tables_by_paper into a list of tables with associated information to create a dataset."""
    dataset = []
    for paper_i, paper in enumerate(tables_by_paper):
        for table_key in paper["tables"]:
            if "soup" in paper["tables"][table_key]:
                table_soup = paper["tables"][table_key]["soup"]
            else:
                table_soup = BeautifulSoup(paper["tables"][table_key]["table_html"])
            cites = table_soup.find_all("cit")

            cite_shas = [cite.get("sha") for cite in cites]

            print(len(dataset), paper_i, table_key)
            dataset.append(
                {
                    "paper_id": paper["paper_id"],
                    "_pdf_hash": paper["_pdf_hash"],
                    "_source_hash": paper["_source_hash"],
                    "_source_name": paper["_source_name"],
                    "_table_hash": table_key,
                    "table_html": str(table_soup),
                    "table_json": soup_to_json(table_soup),  # this is kinda hard
                    "bib_hash": cite_shas,
                }
            )
    return dataset


def main():
    argp = ArgumentParser()
    argp.add_argument("in_path", type=str)
    argp.add_argument("out_path", type=str)
    args = argp.parse_args()

    valid_tables = extract_valid_tables(args.out_path, DEFAULT_TABLE_FILTERS)
    valid_tables_dataset = create_dataset(valid_tables)
    with open(args.out_path, "w") as f:
        for sample in valid_tables_dataset:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
