import os
import csv

for split in ["train", "test"]:
    data_dir = f"/mnt/chrastil/users/marjanrsd/openbhb_fsoutput/{split}"
    sub_dirs = os.listdir(data_dir)
    sub_ids = []
    hemispheres = ["rh", "lh"]
    for d in sub_dirs:
        if len(d) == 4:
            try:
                int(d)
                # add if it can be interpreted as an int
                sub_ids.append(d)
            # if it can't, catch error and continue loop
            except:
                continue
    # overwrite sub_dirs with filtered paths
    sub_dirs = []
    for i in sub_ids:
        sd = os.path.join(data_dir, i)
        sub_dirs.append(sd)
    sub_dirs.sort()

    thick_rows = []
    for sub_dir in sub_dirs:
        print("\nsub_dir: ", sub_dir)
        thick_row = []
        sub_id = sub_dir[-4:]
        t1_path = f"{split}/{sub_id}_T1.npy"
        thick_row.append(t1_path)
        for hem in hemispheres:
            print("\nhemisphere: ", hem)
            hem_stats = f'{hem}.aparc.stats'
            stats_file = os.path.join(sub_dir, 'stats/', hem_stats)

            try:
                with open(stats_file, "r+") as f:
                    lines = f.readlines()
            except:
                continue # e.g. don't have data for this pt
    
            roi_names = [
                "bankssts",                                 
                "caudalanteriorcingulate",                  
                "caudalmiddlefrontal",                     
                "cuneus",                                   
                "entorhinal",                                
                "fusiform",                                 
                "inferiorparietal",                         
                "inferiortemporal",                         
                "isthmuscingulate",                         
                "lateraloccipital",                         
                "lateralorbitofrontal",                     
                "lingual",                                  
                "medialorbitofrontal",                      
                "middletemporal",                           
                "parahippocampal",                          
                "paracentral",                              
                "parsopercularis",                          
                "parsorbitalis",                            
                "parstriangularis",                         
                "pericalcarine",                            
                "postcentral",                              
                "posteriorcingulate",                       
                "precentral",                               
                "precuneus",                                
                "rostralanteriorcingulate",                 
                "rostralmiddlefrontal",                     
                "superiorfrontal",                         
                "superiorparietal",                         
                "superiortemporal",                         
                "supramarginal",                            
                "frontalpole",                               
                "temporalpole",                              
                "transversetemporal",                        
                "insula"                 
            ]
            for roi_name in roi_names:
                print("ROI: ", roi_name)
                for l in lines:
                    if roi_name in l:
                        # edge case!
                        if roi_name == "cuneus":
                            if "precuneus" in l:
                                continue
                        split_l = l.split(" ")
                        # filter blank/empty list elements
                        split_l = [x for x in split_l if x != '']
                        thick_avg = split_l[4]
                        thick_row.append(thick_avg)
                        print("avg thickness: ", thick_avg)
        if len(thick_row) < 69:
            continue # empty cells. skip.
        thick_rows.append(thick_row)

    filename = os.path.join(os.getcwd(), f"{split}.csv")
    col_header = []
    for h in hemispheres:
        for r in roi_names:
            col_header.append(h + ' ' + r)
    col_header = ["sub_ID"] + col_header
    with open(filename, "w") as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(col_header)
        csv_writer.writerows(thick_rows)
