import spacy
# Load the pre-trained NER model


nlp_ner = spacy.load("nlp/model/model-best")
def ner(text):
# Process the text with the NER model
    #doc = nlp_ner("TAX INVOICE KIAH INTERNATIONAL K-1 APMC MKT-II, PHASE - II, DANABANDAR, VASHI, NAVI MUMBAI - 400705 PAN NO: MUMJ20382F PHONE : 9137727033 GSTIN : 27AJEPM2071L1Z2 FSSAI : 11516016000015 DATE 07/01/2026 41F/VASHI/2016/COMP/LICNO.73(W.H). ACK No : 122317020018997 IRN : bc5d819aba8548e0a84f33a9a5c4b83aaa6b1954d62f8307cc0dceed50c32ca1 M/s.AVENUE SUPERMARTS LTD INVOICE NO. :4221 AVENUE SUPERMARTS LTD SR NO 52 HISSA NO 1/1 DATTAWADI ROAD DATE : 08/06/2023 BEHIND BLUE ART NEREGAON DIST.PUNE 411052 PUNE-DUTTAWADI D.O.No. : 0 Del Dt : 08/06/2023 BROKER : NARESH VASANJI SANGOI LORRY : - STATE : 27 MAHARASHTRA FSSAI 11518036001657 28/12/24 Transport : GSTIN : 27AACCA8432H1ZQ PAN : AACCA8432H Remark : Sr Description of Goods HSN Code GST % Qty Net Weight Rate Disc. Amount 1 BHAGAR (30KG)-KIAH 10089090 0.00 67 2010.000 10100.00 4060.20 198949.80 OUR BANK DETAIL ICICI BANK LTD.-VASHI IFSC CODE - ICIC0000228 A/C NO-022805002126 RS. TWO LAKH FIVE HUNDRED FORTY ONE ONLY TOTAL: 67 2010.000 4060.20 198949.80 DELIVERY AT BARDAN / PACKING 0.00 SUBJECT TO NAVI MUMBAI JURISDICTION GODOWN APMC/NMMC CHARGES 1591.60 Certified that the Particulars given above are true and correct TAXABLE TOTAL 0.00 Electronic Reference Number : K-1 GST AMOUNT Tax Is Payable On Reverse Charge : (Y/N) : NO Amount of Tax Subject to Reverse charge E. & O.E 0.00 ROUND OFF -0.40 CGST SGST IGST INVOICE TOTAL ₹ 200541.00 Rate Amount Rate Amount Rate Amount For : KIAH INTERNATIONAL I/We hereby certify that food/foods mentioned in the invoice is/are warranted to be of the nature and quality which it/ these purports /purport to be")
    doc = nlp_ner(text)
    # Extract information from named entities
    extracted_info = []
    for ent in doc.ents:
        entity_info = {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
            "extracted_text": doc[ent.start:ent.end].text
        }
        extracted_info.append(entity_info)

    return extracted_info
