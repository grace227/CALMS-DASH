# %%
import fitz  # PyMuPDF


def add_highlights_to_pdf(input_pdf, output_pdf, terms_to_highlight):
    doc = fitz.open(input_pdf)
    for page in doc:
        text_instances = [page.search_for(term) for term in terms_to_highlight]
        for instance in text_instances:
            for inst in instance:
                highlight = page.add_highlight_annot(inst)
    doc.save(output_pdf)
    doc.close()

# add_highlights_to_pdf("original.pdf", "highlighted.pdf", ["term1", "term2"])


# %%
input_pdf = '/home/beams/YLUO89/gpts/CALMS-DASH/assets/Chen et al 2014.pdf'
output_pdf = '/home/beams/YLUO89/gpts/CALMS-DASH/assets/Chen et al 2014_mod.pdf'
terms_to_highlight = 'We have also shown the capabilities of the BNP for examining frozen-hydrated whole cells at sub-50 nm spatial resolution. The cryogenic capabilities of the BNP, along with its very high spatial reso- lution, hard X-ray capabilities, high-throughput and motion stability will enable us to address a wide range of challenges in life sciences and biological research'

add_highlights_to_pdf(input_pdf, output_pdf, terms_to_highlight)

# %%



