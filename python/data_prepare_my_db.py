import click

from data_load import load_mitbih_db


@click.command()
@click.option("--whole", default=0, prompt="0 or 1, 1 stands for all kinds of my_db",
              help="0 or 1, 1 stands for all kinds of my_db")
def prepare_db(whole=0):
    if not whole:
        load_mitbih_db("DS1",
                       False,
                       ws=[90, 90],
                       do_preprocess=True)
        load_mitbih_db("DS2",
                       False,
                       ws=[90, 90],
                       do_preprocess=True)
    else:
        for DS in ["DS1", "DS2"]:
            for is_reduce in (True, False):
                for do_preprocess in (True, False):
                    print("make db for DS({}), is_reduce({}), do_preprocess({})".format(DS, is_reduce, do_preprocess))
                    load_mitbih_db(DS,
                                   is_reduce,
                                   ws=[90, 90],
                                   do_preprocess=do_preprocess)


if __name__ == "__main__":
    prepare_db()
