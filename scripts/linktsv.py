import logging;
from linking.wikidata_linker import WikidataLinker

sourcefile = "/home/mireleschavezv/Downloads/ner_in_cxt_de.tsv"
outfile = "/home/mireleschavezv/Downloads/ner_in_cxt_de_linked.tst"

dl = WikidataLinker(language="de")

lastent = None
lastctxt = None
lasturi = None
with open(sourcefile) as fin:
    with open(outfile, "w") as fout:
        for ln,line in enumerate(fin):
            if ln==0:
                continue
            lines=line.strip().split("\t")
            ent = lines[0].split("::")[0]
            context = lines[3]
            start_off = int(lines[1])
            end_off = int(lines[2])
            uri = str(dl.link_within_context(surface_form=ent,
                                             start_offset=start_off,
                                             end_offset=end_off,
                                             context=context))
            fout.write("\t".join(lines+[uri])+" \n")
            #print(ent,uri,ln)
            if lastent==ent and lasturi != uri:
                print(ent)
                print(lasturi,"\t", lastctxt,"\n\t!")
                print(uri,"\t", context)
                print("\t-------",ln,"\n")
            lastent=ent
            lastctxt=context
            lasturi = uri
        if ln % 100 == 99:
            dl._write_cache()
            fout.flush()



