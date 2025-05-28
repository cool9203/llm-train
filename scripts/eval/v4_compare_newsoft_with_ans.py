import json

import pandas as pd

try:
    import cn2an

    support_cn2an = True
except ImportError:
    support_cn2an = False

print(f"是否支援中文數字轉換: {support_cn2an}")


def get_json_result(json_input_path):
    with open(json_input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def v4_compare_with_ans(ans_csv_file, api_result_path, compare_output_path):
    df = pd.read_csv(ans_csv_file, encoding="utf-8", dtype=str)
    api_result = get_json_result(api_result_path)
    # results = [{'filename': '_1212_231227085500552_007_9C82033B.jpg', 'api_result': {'result': {'InvoiceNumber': {'value': 'TY37092609'}, 'CompanyName': {'value': '旭聖光電企業有限公司'}, 'VnUniformNumber': {'value': '84172879'}, 'BuUniformNumber': {'value': '28752380'}, 'InvoiceDate': {'value': '2023-11-01'}, 'ItemName': [{'value': 'AP2F90V0059'}], 'NetAmount': {'value': '600'}, 'TaxAmount': {'value': '300'}, 'TotalAmount': {'value': '6300'}, 'TotalAmountCH': {'value': '陸仟叁佰元整'}}, 'processing_time': '4.76s', 'id': 'b2b64dc97c12450555b4ebb51ef2e61e'}}, {'filename': '_1212_231227085501301_008_F5C749F9.jpg', 'api_result': {'result': {'InvoiceNumber': {'value': 'TY37092615'}, 'CompanyName': {'value': '旭望光電企業有限公司'}, 'VnUniformNumber': {'value': '84172879'}, 'BuUniformNumber': {'value': '28752380'}, 'InvoiceDate': {'value': '2023-11-01'}, 'ItemName': [{'value': 'AP-20-FD02940+RL+T1'}, {'value': 'AP-20-FD02940+RL+DEC'}, {'value': 'AP-20-FD02940+POS+RL+T1'}], 'NetAmount': {'value': '39660'}, 'TaxAmount': {'value': '1983'}, 'TotalAmount': {'value': '41643'}, 'TotalAmountCH': {'value': ' 肆萬壹仟陸佰肆拾叁元'}}, 'processing_time': '6.19s', 'id': 'efa92a1e597cfe3a1a7139cc9f12e42f'}}, {'filename': '_1212_231227085502260_009_C90378AC.jpg', 'api_result': {'result': {'InvoiceNumber': {'value': 'TY37092607'}, 'CompanyName': {'value': '旭壓光電企業有限公司'}, 'VnUniformNumber': {'value': '84172879'}, 'BuUniformNumber': {'value': '28752380'}, 'InvoiceDate': {'value': '2023-11-01'}, 'ItemName': [{'value': '電路板'}], 'NetAmount': {'value': '4500'}, 'TaxAmount': {'value': '225'}, 'TotalAmount': {'value': '4725'}, 'TotalAmountCH': {'value': '肆仟柒佰贰拾伍元'}}, 'processing_time': '4.64s', 'id': 'f5088d04806931ff5bf12b2023704cdc'}}]
    # api_result = [{
    #     "filename": "_1212_231227085502260_009_C90378AC.jpg",
    #     "api_result": {
    #         "InvoiceNumber": "TY37092607",
    #         "CompanyName": "協麟光電企業有限公司",
    #         "VnUniformNumber": "84172879",
    #         "BuUniformNumber": "28752380",
    #         "InvoiceDate": "中華民國112年11月1日",
    #         "item": [
    #             "電路板"
    #         ],
    #         "NetAmount": "4500",
    #         "TaxAmount": "225",
    #         "TotalAmount": "4725",
    #         "TotalAmountCH": "肆仟柒佰貳拾伍元"
    #     },
    #     "used_time": 6.782865524291992
    # }]
    # 跟正確答案比較
    global InvoiceNumber_counter
    InvoiceNumber_counter = 0
    global CompanyName_counter
    CompanyName_counter = 0
    global VnUniformNumber_counter
    VnUniformNumber_counter = 0
    global BuUniformNumber_counter
    BuUniformNumber_counter = 0
    global InvoiceDate_counter
    InvoiceDate_counter = 0
    global NetAmount_counter
    NetAmount_counter = 0
    global TaxAmount_counter
    TaxAmount_counter = 0
    global TotalAmount_counter
    TotalAmount_counter = 0
    global TotalAmountCH_counter
    TotalAmountCH_counter = 0

    all_ary = []
    for item in api_result:
        single_dict = {}
        filename = item["filename"]
        # item_ocr_result =  item['api_result']['result']
        item_ocr_result = item["api_result"]
        matched_row = df[df["filename"] == filename]
        for match_row_key in matched_row:
            if match_row_key == "filename":
                single_dict[match_row_key] = filename
                continue
            # print(match_row_key, matched_row[match_row_key])
            # print(item_ocr_result[match_row_key]['value'], matched_row[match_row_key].iloc[0])
            # if filename not in all_dict:
            #     all_dict[filename] = {}
            if match_row_key in item_ocr_result:
                single_dict[f"{match_row_key}_iii"] = str(item_ocr_result[match_row_key])
                single_dict[f"{match_row_key}_ans"] = str(matched_row[match_row_key].iloc[0])

                if str(item_ocr_result[match_row_key]) == str(matched_row[match_row_key].iloc[0]):
                    globals()[f"{match_row_key}_counter"] += 1
                    single_dict[f"{match_row_key}_result"] = 1
                else:
                    if match_row_key == "InvoiceDate":
                        if str(item_ocr_result[match_row_key]) == f"中華民國{str(matched_row[match_row_key].iloc[0])}":
                            globals()[f"{match_row_key}_counter"] += 1
                            single_dict[f"{match_row_key}_result"] = 1
                            continue
                    if match_row_key == "TotalAmountCH" and support_cn2an:
                        try:
                            if cn2an.cn2an(
                                str(item_ocr_result[match_row_key]).replace("參", "叁").replace("兩", "二")
                            ) == cn2an.cn2an(str(matched_row[match_row_key].iloc[0]).replace("參", "叁").replace("兩", "二")):
                                globals()[f"{match_row_key}_counter"] += 1
                                single_dict[f"{match_row_key}_result"] = 1
                            continue
                        except ValueError:
                            pass

                    single_dict[f"{match_row_key}_result"] = 0
            else:
                # if filename not in all_dict:
                #     all_dict[filename] = {}
                print(match_row_key, item_ocr_result)
                single_dict[f"{match_row_key}_iii"] = f"沒有這個key值: {match_row_key}"
                single_dict[f"{match_row_key}_ans"] = str(matched_row[match_row_key].iloc[0])
                single_dict[f"{match_row_key}_result"] = 0
        all_ary.append(single_dict)
    # print(all_ary)
    with open(f"{compare_output_path}.json", "w", encoding="utf-8") as json_file:
        json.dump(all_ary, json_file, ensure_ascii=False, indent=4)
    print(
        {
            "InvoiceNumber_counter": InvoiceNumber_counter,
            "CompanyName_counter": CompanyName_counter,
            "VnUniformNumber_counter": VnUniformNumber_counter,
            "BuUniformNumber_counter": BuUniformNumber_counter,
            "InvoiceDate_counter": InvoiceDate_counter,
            "NetAmount_counter": NetAmount_counter,
            "TaxAmount_counter": TaxAmount_counter,
            "TotalAmount_counter": TotalAmount_counter,
            "TotalAmountCH_counter": TotalAmountCH_counter,
        }
    )

    # 將結果轉換為 DataFrame
    results_df = pd.DataFrame(all_ary)
    results_df.to_csv(f"{compare_output_path}.csv", index=False, encoding="utf-8")
    results_df.to_excel(f"{compare_output_path}.xlsx", index=False, engine="openpyxl")
    # 比對結果
    # comparison = pd.merge(df, results_df, on='filename', how='outer')

    # 顯示比對結果
    # print(comparison)


ans_csv = "result_final.csv"
# ans_csv = "newsoft_handwriting_20250416_new_format_remove_duplicate.csv"
# api_result_path = "yoga_5005pic_6000steps_v4_api_result.json"
# api_result_path = "yoga_5005pic_6000steps_v4_api_test803pics_result.json"
# api_result_path = "pack_test_result.json"
# api_result_path = "trl_test100pics_7b_max_pixel_result.json"
# api_result_path = "trl_test100pics_3b_0526_result.json"
# api_result_path = "trl_test100pics_3b_0527_max_pixel_ck-7425_result.json"
api_result_path = "trl_test100pics_3b_0527_ck-4455_result.json"


# compare_result_output = 'yoga_5005pic_6000steps_v4_compare_result'
# compare_result_output = 'yoga_5005pic_6000steps_v4_api_test803pics_compare_result'
# compare_result_output = 'pack_test_compare_result'
# compare_result_output = 'trl_test100pics_7b_max_pixel_compare_result'
compare_result_output = "trl_test100pics_3b_0527_ck-4455_compare_result"


v4_compare_with_ans(ans_csv, api_result_path, compare_result_output)
