from argparse import ArgumentParser
import functools
import gzip
import json
import multiprocessing
import os
import re
import sys
from typing import List
import warnings

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


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


def not_too_long_15e3(table_soup):
    # return len(str(table_soup)) < 5e3
    return len(str(table_soup)) < 15e3


def not_too_long_5e3(table_soup):
    return len(str(table_soup)) < 5e3


def not_too_long_or_short(table_soup):
    # return len(str(table_soup)) < 5e3
    return 398 < len(str(table_soup)) < 15e3


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
    return len(table_soup.find_all("cit")) >= 2


def has_cites_in_first_row_or_col(table_soup):
    """
    Checks for citations in the first row or column
    (this is quite restrictive)
    """
    min_num_cites = 2
    trs = table_soup.find_all("tr")

    # check the first, non-empty row
    i = 0
    while i < len(trs):
        first_row = trs[i]
        cells = first_row.find_all("td")

        # skip any all-empty rows
        if all([not cell.text.strip() for cell in cells]):
            i += 1
            continue

        if len(first_row.find_all("cit")) >= min_num_cites:
            return True
        else:
            break

    # check the first column (usually not empty)
    # first_col_citations = [row.find("cit") for row in trs if row.find("cit") is not None]
    first_col_citations = [row.find_all("td")[0].find("cit") for row in trs]
    return len([cell for cell in first_col_citations if cell is not None]) >= min_num_cites


def has_cites_in_rows_or_cols(table_soup):
    """
    Checks *any* row or col to see if it has >2 cites
    """
    min_num_cites = 2
    trs = table_soup.find_all("tr")

    valid_table = False
    max_num_cells = 0
    for row in trs:
        cells = row.find_all("td")
        # skip any all-empty rows
        if all([not cell.text.strip() for cell in cells]):
            continue

        # track max number of cells per row to help with processing column
        max_num_cells = max(max_num_cells, len(cells))

        if len(row.find_all("cit")) >= min_num_cites:
            valid_table = True
            break

    # check columns. Assumes that we don't have weird multicolumn stuff going on,
    # so it's a bit coarse. E.g. row[0][3] is the same column as row[3][3]
    if not valid_table and max_num_cells > 0:
        for col_i in range(max_num_cells):
            col_citations = []
            for row in trs:
                try:
                    col_citations.append(row.find_all("td")[col_i].find("cit"))
                except IndexError:
                    continue
            # print(col_citations)
            if len([cell for cell in col_citations if cell is not None]) >= min_num_cites:
                valid_table = True
                break
    return valid_table


def has_max_one_cite_per_cell(table_soup):
    """
    We don't want any tables with more than one citation per cell.
    Returns True if any cell has a maximum of one citation
    """
    max_cites_per_cell = 1
    trs = table_soup.find_all("tr")

    valid_table = False
    for row in trs:
        for cell in row.find_all("td"):
            if len(cell.find_all("cit")) > max_cites_per_cell:
                return False
    return True


# FLOAT_REGEX = re.compile("\d\.\d")
FLOAT_REGEX = re.compile("\.\d")


def has_no_floats(table_soup):
    for cell in table_soup.find_all("td"):
        cell_text = cell.get_text()
        if cell_text and FLOAT_REGEX.search(cell_text) is not None:
            return False

    # some tables have floats in paragraphs, which is confusing but
    # not always recovered for some reason
    for cell in table_soup.find_all("p"):
        cell_text = cell.get_text()
        if cell_text and FLOAT_REGEX.search(cell_text) is not None:
            return False
    # for s in table_soup.strings:
    #     s = s.strip()
    #     if s and FLOAT_REGEX.search(s) is not None:
    #         return False
    return True


def has_table_cells(table_soup):
    return table_soup.find("td") is not None


def has_no_figures(table_soup):
    for cell in table_soup.find_all("td"):
        cell_text = cell.get_text()
        if cell_text and r"{{figure" in cell_text:
            return False

    for cell in table_soup.find_all("p"):
        cell_text = cell.get_text()
        if cell_text and r"{{figure" in cell_text:
            return False
    return True


DEFAULT_TABLE_LABELS = [
    not_too_long_15e3,
    not_too_long_or_short,
    has_table_cells,
    has_at_least_2_cites,
    has_max_2_sub_tables,
    has_at_least_2_cols,
    has_at_least_2_rows,
    has_no_floats,
    has_no_figures,
    has_cites_in_rows_or_cols,
    has_cites_in_first_row_or_col,
    has_max_one_cite_per_cell,
]

DEFAULT_TABLE_FILTERS = [
    has_cites_in_rows_or_cols,
    has_at_least_2_cites,
    has_no_floats,
    not_too_long_or_short,
    has_at_least_2_cols,
    has_max_2_sub_tables,
    has_at_least_2_rows,
    has_no_figures,
    has_table_cells,
]

COLORS = r"((alice)?blue|black|(mid)?gr[ae]y|red|(dark)?green)"
COLORS_RE = rf"{COLORS}(\!\d\d?|(?=[✗✓]))"


def is_na(text):
    return text.lower() == "n/a" or not text.strip() or text == "\u2216"


def extract_valid_tables(path, table_filters, label_tables=False):
    """
    If `label_tables` is True, return all of the tables, but list
    what labels are true/false for each of them.


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
    if os.path.splitext(path)[1] == ".gz":
        f = gzip.open(path, "r")
    else:
        f = open(path)
    with f:
        added_tables = set()
        for line in tqdm(f):
            paper = json.loads(line)
            # if paper["paper_id"] != "2310.03103v1":
            #     continue
            filtered_tables = {}
            # print(len(paper["tables"]))
            for key, table in paper["tables"].items():
                if table["table"]:
                    # print("Preparing soupification")
                    table_soup = soupify(table["table"])

                    if label_tables:
                        # label the tables with the filters they pass
                        labels = {}
                        for flter in table_filters:
                            labels[flter.__name__] = flter(table_soup)
                        labels["len"] = len(str(table_soup))
                    else:
                        # Filter tables
                        exit_early = False
                        for flter in table_filters:
                            if not flter(table_soup):
                                # exit early as soon as a filter is wrong
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
                    if label_tables:
                        filtered_tables[key]["labels"] = labels

            if filtered_tables:
                new_paper = {k: v for k, v in paper.items() if k != "tables"}
                new_paper["tables"] = filtered_tables
                valid_tables.append(new_paper)

            # For debugging
            # if len(valid_tables) > 10:
            #     break

    return valid_tables


def split_references_column(table_df):
    # next, break the citation into their own column with the heading "References"
    # this new References column will be the *index* of the dataframe, so all it's elements
    # must be unique

    # if "Type of deformations" in table_df.columns:
    #     breakpoint()
    # determine which column has the most references

    # use iloc in case the name of the column is repeated
    column_with_cites = table_df.iloc[:, 0]  # by default it's the first one
    max_num_cites = -1
    for col_i, _ in enumerate(table_df.columns):
        num_cites = 0
        for cell_val in table_df.iloc[:, col_i]:
            matches = re.search("{{cite:[a-f\d]{7}}}", cell_val)
            if matches is not None:
                num_cites += 1
        if num_cites > max_num_cites:
            max_num_cites = num_cites
            column_with_cites = table_df.iloc[:, col_i]

    if isinstance(column_with_cites, pd.DataFrame):
        column_with_cites = column_with_cites.agg("".join, axis=1)
    references_col = []
    # try:
    new_column_without_cites_name = column_with_cites.name
    # except AttributeError:
    #    breakpoint()
    new_column_without_cites = []
    no_cite_count = 0
    for cell_val in column_with_cites:
        matches = re.search("{{cite:[a-f\d]{7}}}", cell_val)
        if matches is None:
            references_col.append(f"no_cite-{no_cite_count}")
            new_column_without_cites.append(cell_val)
            no_cite_count += 1
        else:
            references_col.append(matches[0])
            new_cell_val = cell_val.replace(matches[0], "")
            if not new_cell_val:
                new_cell_val = "-"
            new_column_without_cites.append(new_cell_val.strip())

    if any([val != "-" for val in new_column_without_cites]):
        if new_column_without_cites_name == "References":
            # This only gets triggered in weird cases where one of the columns is "References" but
            # we don't successfully parse out all of the citations (eg if there is more than one cite)
            # in the column
            table_df = table_df.rename(columns={new_column_without_cites_name: "References_OLD"})
            table_df["References_OLD"] = new_column_without_cites
        else:
            table_df[new_column_without_cites_name] = new_column_without_cites
        try:
            table_df.insert(0, "References", references_col)
        except ValueError:
            print("Error inserting into table_df")
            raise ValueError
    else:
        # if the column just has citations, rename the column
        table_df = table_df.rename(columns={new_column_without_cites_name: "References"})
        # and reorder them so that "References" is first
        reordered_cols = ["References"] + [col for col in table_df.columns if col != "References"]
        table_df = table_df[reordered_cols]
    assert table_df.columns[0] == "References"
    # if table_df.columns[0] != "References":
    #     breakpoint()
    return table_df


def postprocess_table_df(table_df):
    """
    Converts a list, where each element is row, into a dictionary representing
    the table. This conversion is done using pandas and then a large amount of
    post-processing this conversion, this method
    """

    # if the citations are in the columns, then change them to the rows
    if " ".join(table_df.columns).count("{{cite:") > 0:
        original_col_0 = table_df.columns[0]
        table_df = table_df.set_index(original_col_0)
        table_df = table_df.transpose()
        try:
            table_df = table_df.reset_index(names=original_col_0)
        except ValueError:
            # something went wrong so... transpose back
            table_df = table_df.transpose().reset_index()
            # breakpoint()

    def process_cell(cell):
        # replace non-breaking space with normal space
        if isinstance(cell, tuple):
            cell = str(cell)
        cell = cell.replace("\u00a0", " ")
        cell = cell.strip()

        # normalize apostrophes
        cell = re.sub("[’]", "'", cell)
        # binary no
        if cell == "X":
            cell = "\u2717"
        # cell = re.sub(f"{COLORS}\u2717", "\u2717", cell)
        # binary yes - standardize
        cell = re.sub("\u2714", "\u2713", cell)

        # remove color annotations from end
        cell = re.sub(f"{COLORS_RE}", "", cell)
        # and from beginning - this is potentially dangerous...
        cell = re.sub(rf"^{COLORS}", "", cell)
        # empty cells should be "-" instead
        cell = re.sub(r"(N/A|none)", "-", cell)
        cell = cell.strip()
        if cell == "":
            cell = "-"
        return cell

    table_df = table_df.map(process_cell)
    table_df.columns = table_df.columns.map(process_cell)
    # print(table_df.columns)

    # if the cells have citations and other information, put the citations into a new cell
    table_df = split_references_column(table_df)
    # TODO: if rows have the same reference, then combine their values into a list
    return table_df


def merge_rows(table, row_i, row_j):
    new_row = []
    for val_i, val_j in zip(table[row_i], table[row_j]):
        if val_i == val_j or not val_j.strip():
            new_row.append(val_i)
        elif not val_i.strip():
            new_row.append(val_j)
        else:
            new_row.append("-".join([val_i, val_j]))
    table[row_i] = new_row
    del table[row_j]
    return table


def soup_to_json(table_soup, verbose=False):
    # first, determine the number of columns as the max number of cells in a row
    num_cols = max(
        [len(row.find_all("td")) for row in table_soup.find_all("tr")]
        + [
            sum([int(cell.attrs.get("cols", "1")) for cell in row.find_all("td")])
            for row in table_soup.find_all("tr")
        ]
    )
    # next, determine the number of rows:
    num_rows = len(table_soup.find_all("tr"))

    if verbose:
        print(num_rows, num_cols)

    # next, extract the values. Some rows are "header" rows and contain explanatory info.
    # The rows we are unable to parse are tracked separately in a "incomplete_rows" field

    table = {"incomplete_rows": [], "table": [["" for ci in range(num_cols)] for ri in range(num_rows)]}
    columns = [[] for _ in range(num_cols)]
    transposed_table = False

    # Next, fill in table[row_i][col_i]
    header_rows = []
    seen_cites = False
    for row_i, row in enumerate(table_soup.find_all("tr")):
        cells = row.find_all("td")

        col_i = 0
        # if a row does not contain any citations and we only have one row in the table, then say it's part of the header row
        num_cites = len([cell for cell in cells if cell.find("cit") is not None])
        if num_cites > 0:
            seen_cites = True
        if num_cites == 0 and row_i <= 1 and not seen_cites:
            if verbose:
                print(f"no cites in row: {row_i}, adding as header")
            header_rows.append(row_i)

        # determine if the current row is a header row
        if len(cells) < num_cols and not seen_cites:
            if verbose:
                print(f"not enough cols in row: {row_i}, adding as header")
            header_rows.append(row_i)
        elif len(cells) < num_cols or any(["{{figure:" in cell.text for cell in cells]):
            # for cell in cells:
            table["incomplete_rows"].append(
                {
                    "row_idx": row_i,
                    "cells": [cell.text for cell in cells],
                }
            )
            continue

        for cell in cells:
            # acount for multi-column cells
            num_spanning_cols = int(cell.attrs.get("cols", "1"))
            for col_offset in range(num_spanning_cols):
                cell_text = cell.text
                # account for multi-row cells
                multirow_cell = re.search(r"(\d)\*(.+)", cell_text)
                if multirow_cell:
                    count = int(multirow_cell[1])
                    cell_text = multirow_cell[2]
                    if verbose:
                        print(count, cell_text)
                    for row_offset in range(1, count):
                        # add this to subsequent rows
                        try:
                            table["table"][row_i + row_offset][col_i + col_offset] += cell_text.strip()
                        except IndexError:
                            print(
                                "Index error assigning value. This probably means we were unable to parse the table correctly. Skipping..."
                            )
                            return {
                                "table": table["table"],
                                "incomplete_rows": table["incomplete_rows"],
                                "table_dict": {},
                            }

                # if a cell contains a horizontal line, don't add it but mark it as a header row
                if re.search("\(r\)\d-\d", cell_text):  #  and not seen_cites:
                    # print("horizontal line added as header_row")
                    # header_rows.append(row_i + 1)
                    cell_text = re.sub("\(r\)\d-\d", "", cell_text)
                    # continue

                if (
                    is_na(cell_text.strip())
                    and not table["table"][row_i][col_i + col_offset]
                    and not row_i in header_rows
                ):
                    cell_text = "-"

                if not table["table"][row_i][col_i + col_offset]:
                    table["table"][row_i][col_i + col_offset] += cell_text.strip()
            col_i += num_spanning_cols

    # after we're done filling in the table, collapse the header row
    # merge the first two rows until we reach the header row
    if verbose:
        print(header_rows)
    if header_rows:
        for _ in range(max(header_rows)):
            table["table"] = merge_rows(table["table"], 0, 1)

    # remove any empty rows
    table_filtered = []
    for row in table["table"]:
        if any(row):
            table_filtered.append(row)
    table["table"] = table_filtered

    if verbose:
        for row in table["table"]:
            print(row)
        # print(table["table"])
    # next, assume the first row has the column headers and the first col has the row headers
    if not any(["".join(row) for row in table["table"]]):
        table_dict = {}
    else:
        table_df = pd.DataFrame(table["table"][1:], columns=table["table"][0])
        table_df = postprocess_table_df(table_df)
        table_dict = table_df.to_dict(orient="list")

        # if len(table_dict) != len(table["table"][0]):
        #     table_dict = {}
    # del table["table"]  # we don't actually need the list of rows, I don't think
    table["table_dict"] = table_dict
    return table


def get_table_row_bib_map(table_json, bib_hashes, paper_id) -> List:
    """
    Uses the heuristic that if a table contains a row that doesn't have a citation,
    then that row represents the containing paper, as long as the cell doesn't contain
    certain words e.g. "standard".

    Returns a List where each element represents a row of the table. Each element contains
    a row number, the corpus id, bib_hash or arxiv id, and whether the row is the paper
    with the table ("ours") or an external reference ("ref").

    TODO: could also be "above". there could also be more than one "ours" row.
    """

    table_row_bib_map = []
    cite_id_map = {bib_ref[:7]: bib_ref for bib_ref in bib_hashes if bib_ref is not None}
    table_df = pd.DataFrame(table_json)
    ours_row = None
    for i, cell_val in enumerate(table_df[table_df.columns[0]]):
        # extract the citation
        matches = re.search("{{cite:([a-f\d]{7})}}", cell_val)
        if matches is None:
            # we could be in an "ours" row
            if "standard" in cell_val.lower():
                # this is
                continue
            else:
                # track the last unmatched row as "ours"
                ours_row = {
                    "bib_hash_or_arxiv_id": paper_id,
                    "row": i,
                    "corpus_id": -1,  # TODO: After running `populate_bib_entries`, this should be replaced with the correct corpus id
                    # bib_entries[table_original["paper_id"]]["corpus_id"],
                    "type": "ours",
                }
        else:
            cite_id = matches[1]
            bib_hash_match = cite_id_map[cite_id]
            table_row_bib_map.append(
                {
                    "bib_hash_or_arxiv_id": bib_hash_match,
                    "row": i,
                    "corpus_id": -1,  # bib_entries[bib_hash_match]["corpus_id"],  # this will get overwritten
                    "type": "ref",
                }
            )

    if ours_row is not None:
        table_row_bib_map.append(ours_row)
    return table_row_bib_map


def create_dataset(labeled_tables, filters):
    """
    Flattens the tables_by_paper into a list of tables with associated information to create a dataset."""
    dataset = []
    if labeled_tables[0].get("labels") is None:
        print("Unable to find labels on the tables. All are being converted to json.")
    missing_bib_hashes = []
    skipped_table_hashes = []
    for table_i, table in tqdm(enumerate(labeled_tables), total=len(labeled_tables)):
        should_skip_table = False

        # there should usually be labels, but there might not be, in which case don't filter anything...
        if table.get("labels") is not None:
            for filter_fn in filters:
                if not table["labels"][filter_fn.__name__]:
                    should_skip_table = True
                    break
        if should_skip_table:
            continue

        if "table_html" not in table:
            table_soup = soupify(table["xml"])
        else:
            table_soup = soupify(table["table_html"])
        cites = table_soup.find_all("cit")
        cite_shas = [cite.get("sha") for cite in cites]

        try:
            with warnings.catch_warnings(action="ignore"):
                table_json = soup_to_json(table_soup)
        except Exception as e:
            table_json = {"table_dict": {}}
            print(f"Error {e}")

        if not table_json["table_dict"]:
            print(f"Skipping {table['_table_hash']} because `soup_to_json` failed")
            skipped_table_hashes.append(table["_table_hash"])
            continue
        # trp = table_requires_paper(table_json)
        row_bib_map = get_table_row_bib_map(table_json["table_dict"], cite_shas, table["paper_id"])
        # print(len(dataset), table_i, table["_table_hash"])

        new_sample = {
            "paper_id": table["paper_id"],
            "_pdf_hash": table.get("_pdf_hash"),
            "_source_hash": table.get("_source_hash"),
            "_source_name": table.get("_source_name"),
            "_table_hash": table["_table_hash"],
        }

        if "caption" in table:
            new_sample["caption"] = table["caption"]

        new_sample |= {
            "table_html": str(table_soup),
            "table_json": table_json,  # this is kinda hard
            "row_bib_map": row_bib_map,
            "bib_hash": cite_shas,
        }

        if "input_papers" in table:
            for row in row_bib_map:
                if not row["bib_hash_or_arxiv_id"] in table["input_papers"]:
                    missing_bib_hashes.append(row["bib_hash_or_arxiv_id"])
                    continue

                paper_info = table["input_papers"][row["bib_hash_or_arxiv_id"]]
                row["corpus_id"] = paper_info["corpus_id"]
                row["title"] = paper_info["title"]
                row["abstract"] = paper_info["abstract"]
        dataset.append(new_sample)

    print(f"Skipped {len(skipped_table_hashes)} tables")
    print(f"E.g. {skipped_table_hashes[:10]} ...")
    print()
    print(f"Missing {len(missing_bib_hashes)} bib_hashes or arxiv_ids")
    print(f"E.g. {missing_bib_hashes[:10]} ...")
    return dataset
    # for paper_i, paper in enumerate(tables_by_paper):
    #     for table_key in paper["tables"]:
    #         if "soup" in paper["tables"][table_key]:
    #             table_soup = paper["tables"][table_key]["soup"]
    #         else:
    #             table_soup = BeautifulSoup(paper["tables"][table_key]["table_html"])

    #         cites = table_soup.find_all("cit")
    #         cite_shas = [cite.get("sha") for cite in cites]

    #         table_json = soup_to_json(table_soup)
    #         if not table_json["table_dict"]:
    #             print(f"Skipping {table_key} because `soup_to_json` failed")
    #             continue
    #         # trp = table_requires_paper(table_json)
    #         row_bib_map = get_table_row_bib_map(table_json["table_dict"], cite_shas, paper["paper_id"])
    #         print(len(dataset), paper_i, table_key)
    #         dataset.append(
    #             {
    #                 "paper_id": paper["paper_id"],
    #                 "_pdf_hash": paper["_pdf_hash"],
    #                 "_source_hash": paper["_source_hash"],
    #                 "_source_name": paper["_source_name"],
    #                 "_table_hash": table_key,
    #                 "table_html": str(table_soup),
    #                 "table_json": table_json,  # this is kinda hard
    #                 # "table_requires_paper": trp,  # whether the the paper containing the table is one of the rows
    #                 "row_bib_map": row_bib_map,
    #                 "bib_hash": cite_shas,
    #             }
    #         )
    # return dataset


def run(in_path, out_labeled_path, out_filtered_path, label_only, filter_only):
    # labeling
    if not filter_only:
        assert out_labeled_path is not None
        labeled_tables = extract_valid_tables(in_path, DEFAULT_TABLE_LABELS, label_tables=True)
        labeled_tables_dataset = []
        for paper_i, paper in enumerate(labeled_tables):
            for table_key in paper["tables"]:
                labeled_tables_dataset.append(
                    {
                        "paper_id": paper["paper_id"],
                        "_pdf_hash": paper["_pdf_hash"],
                        "_source_hash": paper["_source_hash"],
                        "_source_name": paper["_source_name"],
                        "_table_hash": table_key,
                        "table_html": str(paper["tables"][table_key]["soup"]),
                        "labels": paper["tables"][table_key]["labels"],
                    }
                )
        with open(out_labeled_path, "w") as f:
            for sample in labeled_tables_dataset:
                f.write(json.dumps(sample) + "\n")

    elif not label_only:
        # assumes that `in_path` has labels
        with open(in_path) as f:
            labeled_tables_dataset = [json.loads(line) for line in f]

    # filtering
    if not label_only:
        assert out_filtered_path is not None
        filtered_tables_dataset = create_dataset(labeled_tables_dataset, DEFAULT_TABLE_FILTERS)

        with open(out_filtered_path, "w") as f:
            for sample in filtered_tables_dataset:
                f.write(json.dumps(sample) + "\n")


BLACK_LIST = [
    "done.log",
    "log.txt",
    "2211.jsonl.gz",
    "2212.jsonl.gz",
    "2306.00000-17847v1.jsonl.gz",
    "2308.00000-16912v1.jsonl.gz",
    "2307.jsonl.gz",
]


def main():
    argp = ArgumentParser()
    argp.add_argument("in_path", type=str)
    argp.add_argument("--out_labeled_path", type=str)
    argp.add_argument("--out_filtered_path", type=str)
    argp.add_argument("--label_only", action="store_true")
    argp.add_argument("--filter_only", action="store_true")
    argp.add_argument("--num_processes", type=int, default=1)
    args = argp.parse_args()

    if args.num_processes == 1:
        run(args.in_path, args.out_labeled_path, args.out_filtered_path, args.label_only, args.filter_only)
    else:
        # assume in_path contains the in_paths we care about
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = []
            for in_path in os.listdir(args.in_path):
                if in_path in BLACK_LIST:
                    print(f"Skipping {in_path}")
                    continue
                out_labeled_path = (
                    os.path.join(args.out_labeled_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_labeled_path is not None
                    else None
                )
                out_filtered_path = (
                    os.path.join(args.out_filtered_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_filtered_path is not None
                    else None
                )
                print("In:", os.path.join(args.in_path, in_path))
                print("out:", out_labeled_path)
                runner = functools.partial(
                    run,
                    os.path.join(args.in_path, in_path),
                    out_labeled_path,
                    out_filtered_path,
                    args.label_only,
                    args.filter_only,
                )
                result = pool.apply_async(runner)
                results.append(result)
            for result in results:
                result.get()
            pool.close()
            pool.join()


# def main():
#     argp = ArgumentParser()
#     argp.add_argument("in_path", type=str)
#     argp.add_argument("out_path", type=str)
#     argp.add_argument("--check_yield", action="store_true")
#     argp.add_argument("--label_tables", action="store_true")
#     args = argp.parse_args()

#     assert not args.check_yield or not args.label_tables

#     valid_tables = extract_valid_tables(args.in_path, DEFAULT_TABLE_FILTERS, label_tables=args.label_tables)
#     if args.check_yield:
#         print("Num papers with valid tables:", len(valid_tables))
#         print("Num valid tables:", sum([len(new_paper["tables"]) for new_paper in valid_tables]))
#         sys.exit(0)

#     if args.label_tables:
#         # save the tables as is wo doing json conversion
#         valid_tables_dataset = []
#         for paper_i, paper in enumerate(valid_tables):
#             for table_key in paper["tables"]:
#                 valid_tables_dataset.append(
#                     {
#                         "paper_id": paper["paper_id"],
#                         "_table_hash": table_key,
#                         "table_html": str(paper["tables"][table_key]["soup"]),
#                         "labels": paper["tables"][table_key]["labels"],
#                     }
#                 )
#     else:
#         valid_tables_dataset = create_dataset(valid_tables)
#     with open(args.out_path, "w") as f:
#         for sample in valid_tables_dataset:
#             f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()

    # for row_i, row in enumerate(table_soup.find_all("tr")):
    #     # First, we want to collapse the header rows that contain the column names
    #     cells = row.find_all("td")

    #     # skip any all-empty rows or rows that just have horizontal lines
    #     if all([not cell.text.strip() for cell in cells]):
    #         continue
    #     if all([re.search("\(r\)\d-\d", cell.text.strip()) for cell in cells]):
    #         continue

    #     col_i = 0
    #     for cell in cells:
    #         # acount for multi-column cells
    #         num_spanning_cols = int(cell.attrs.get("cols", "1"))
    #         for col_offset in range(num_spanning_cols):
    #             cell_text = cell.text
    #             # account for multi-row cells
    #             multirow_cell = re.search(r"(\d)\*(.+)", cell_text)
    #             if multirow_cell:
    #                 count = int(multirow_cell[1])
    #                 cell_text = multirow_cell[2]
    #                 print(count, cell_text)
    #                 for row_offset in range(1, count):
    #                     # add this to subsequent rows
    #                     table["table"][row_i + row_offset][col_i + col_offset] += cell_text.strip()

    #             # print(col_i, i)
    #             # if a cell contains a horizontal line, don't add it but mark it as a header row
    #             if re.search("\(r\)\d-\d", cell_text):
    #                 header_rows.append(row_i)
    #                 continue

    #             table["table"][row_i][col_i + col_offset] += cell_text.strip()
    #         col_i += num_spanning_cols

    #     # print(len(cells))
    #     # print(cells)
    #     if len(cells) < num_cols:
    #         # incomplete rows have fewer than the max number of cells. They'll be stored
    #         # in a separate place or sometimes merged into the column headers
    #         if not any(table["table"][0]):
    #             # we haven't started adding to the table, so this is most likely still a header row
    #             if verbose:
    #                 print(cells)
    #             last_col = 0
    #             for cell in cells:
    #                 num_spanning_cols = int(cell.attrs.get("cols", "1"))
    #                 for i in range(last_col, last_col + num_spanning_cols):
    #                     columns[i].append(cell.text)
    #                 last_col += num_spanning_cols
    #             continue

    #         for cell in cells:
    #             table["incomplete_rows"].append(
    #                 {
    #                     "row_idx": row_i,
    #                     "text": cell.text,
    #                     "cols": int(cell.attrs.get("cols", "1")),
    #                 }
    #             )
    #     else:
    #         if not table["table"]:
    #             # column headers
    #             num_cites = len([cell for cell in cells if cell.find("cit") is not None])
    #             print(num_cites, num_cols)
    #             if num_cites >= num_cols - 2:
    #                 if verbose:
    #                     print("table is transposed")
    #                 transposed_table = True
    #             for i, cell in enumerate(cells):
    #                 if cell.text.strip():
    #                     columns[i].append(cell.text)
    #             columns = ["-".join(col).strip() for col in columns]
    #             table["table"].append(columns)
    #         else:
    #             # this is either a row of the table or part of the header
    #             next_row = [cell.text for cell in cells]
    #             num_cites = len([cell for cell in cells if cell.find("cit") is not None])
    #             # if it doesn't contain a citation and we only have one row in the table, then lets say it's part of the header
    #             if num_cites == 0 and len(table["table"]) == 1 and not transposed_table:
    #                 for i, colname_pt2 in enumerate(next_row):
    #                     table["table"][0][i] += f" {colname_pt2}"
    #                     table["table"][0][i] = table["table"][0][i].strip()
    #             else:
    #                 # replace "N/A" and " " with "-"
    #                 next_row = ["-" if is_na(cell_text) else cell_text.strip() for cell_text in next_row]

    #                 next_table_row = []
    #                 for col_i, cell_text in enumerate(next_row):
    #                     if is_na(cell_text):
    #                         next_table_row.append("-")

    #                     matched_multirow_groups = re.search(r"(\d)\*(.+)", cell_text)
    #                     if matched_multirow_groups is not None:
    #                         count = int(matched_multirow_groups[0])
    #                         next_table_row.append(matched_multirow_groups[1])
    #                         for i in range(count - 1):
    #                             # add this to the next row
    #                             [col_i].append(matched_multirow_groups[1])
    #                     elif "":
    #                         next_table_row
    #                     else:
    #                         next_table_row.append(cell_text.strip())

    #                 table["table"].append(next_table_row)

    # OLD implementation
    # table = {"incomplete_rows": [], "table": []}
    # columns = [[] for _ in range(num_cols)]
    # transposed_table = False
    # for row_i, row in enumerate(table_soup.find_all("tr")):
    #     # First, we want to collapse the header rows that contain the column names
    #     cells = row.find_all("td")

    #     # skip any all-empty rows
    #     if all([not cell.text.strip() for cell in cells]):
    #         continue
    #     # print(len(cells))
    #     # print(cells)
    #     if len(cells) < num_cols:
    #         # incomplete rows have fewer than the max number of cells. They'll be stored
    #         # in a separate place or sometimes merged into the column headers
    #         if not (table["table"]):
    #             # we haven't started adding to the table, so this is most likely still a header row
    #             if verbose:
    #                 print(cells)
    #             last_col = 0
    #             for cell in cells:
    #                 num_spanning_cols = int(cell.attrs.get("cols", "1"))
    #                 for i in range(last_col, last_col + num_spanning_cols):
    #                     columns[i].append(cell.text)
    #                 last_col += num_spanning_cols
    #             continue

    #         for cell in cells:
    #             table["incomplete_rows"].append(
    #                 {
    #                     "row_idx": row_i,
    #                     "text": cell.text,
    #                     "cols": int(cell.attrs.get("cols", "1")),
    #                 }
    #             )
    #     else:
    #         if not table["table"]:
    #             # column headers
    #             num_cites = len([cell for cell in cells if cell.find("cit") is not None])
    #             print(num_cites, num_cols)
    #             if num_cites >= num_cols - 2:
    #                 if verbose:
    #                     print("table is transposed")
    #                 transposed_table = True
    #             for i, cell in enumerate(cells):
    #                 if cell.text.strip():
    #                     columns[i].append(cell.text)
    #             columns = ["-".join(col).strip() for col in columns]
    #             table["table"].append(columns)
    #         else:
    #             # this is either a row of the table or part of the header
    #             next_row = [cell.text for cell in cells]
    #             num_cites = len([cell for cell in cells if cell.find("cit") is not None])
    #             # if it doesn't contain a citation and we only have one row in the table, then lets say it's part of the header
    #             if num_cites == 0 and len(table["table"]) == 1 and not transposed_table:
    #                 for i, colname_pt2 in enumerate(next_row):
    #                     table["table"][0][i] += f" {colname_pt2}"
    #                     table["table"][0][i] = table["table"][0][i].strip()
    #             else:
    #                 # replace "N/A" and " " with "-"
    #                 next_row = ["-" if is_na(cell_text) else cell_text.strip() for cell_text in next_row]

    #                 next_table_row = []
    #                 for col_i, cell_text in enumerate(next_row):
    #                     if is_na(cell_text):
    #                         next_table_row.append("-")

    #                     matched_multirow_groups = re.search(r"(\d)\*(.+)", cell_text)
    #                     if matched_multirow_groups is not None:
    #                         count = int(matched_multirow_groups[0])
    #                         next_table_row.append(matched_multirow_groups[1])
    #                         for i in range(count - 1):
    #                             # add this to the next row
    #                             [col_i].append(matched_multirow_groups[1])
    #                     elif "":
    #                         next_table_row
    #                     else:
    #                         next_table_row.append(cell_text.strip())

    #                 table["table"].append(next_table_row)
