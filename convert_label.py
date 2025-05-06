import json
import os

LABEL_PATH = "./Dataset/raw/"
OUTPUT_PATH = "./Dataset/out/"

def convert_one_label(label : json):
    j = json.load(open(LABEL_PATH + label))
    # print(j[0])
    file_name = j[0]['data_title']
    data_hash = j[0]['data_hash']
    frames = j[0]['data_units'][data_hash]['labels'].keys()
    print(frames)
    to_write = {"labels": {}}
    for k in frames:
        list_to_add = []
        
        polygon_number = len(j[0]['data_units'][data_hash]['labels'][k]['objects'])
        # print(polygon_number)
        for i in range(polygon_number):
            key_dict = dict()
            polygon_name = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['name']
            
            if polygon_name != "Instrument Shaft Lavel" and polygon_name != "No Interaction":
                # print(polygon_name)
                key_dict['is_tti'] = 1
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                interaction_type = None
                interaction_tool = None
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    interaction_type = j[0]['object_answers'][object_hash]['classifications'][0]['answers'][0]['name']
                    interaction_tool = j[0]['object_answers'][object_hash]['classifications'][1]['answers'][0]['name']
                key_dict['interaction_type'] = interaction_type
                key_dict['interaction_tool'] = interaction_tool
                key_dict['tti_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
            
            elif polygon_name == "No Interaction":
                key_dict['is_tti'] = 0
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                non_interaction_tool = None
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    non_interaction_tool = j[0]['object_answers'][object_hash]['classifications'][0]['answers'][0]['name']
                key_dict['non_interaction_tool'] = non_interaction_tool
                key_dict['tti_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
            
            else:
                object_hash = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['objectHash']
                instrument_type = None
                if len(j[0]['object_answers'][object_hash]['classifications']) > 0:
                    instrument_type = j[0]['object_answers'][object_hash]['classifications'][0]['answers'][0]['name']
                key_dict['instrument_polygon'] = j[0]['data_units'][data_hash]['labels'][k]['objects'][i]['polygon']
                key_dict['instrument_type'] = instrument_type
            
            
            
            list_to_add.append(key_dict)
            
            # print(key_dict)
            # print(j[0]['data_units'][data_hash]['labels'][k]['objects'][i])
            
            
        to_write['labels'][k] = list_to_add
    
    with open(OUTPUT_PATH+file_name[:-4]+'.json','w') as file:
        json.dump(to_write,file,indent=2)
    


if __name__ == "__main__":
    labels = os.listdir(LABEL_PATH)
    for l in labels:
        convert_one_label(l)


