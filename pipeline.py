import os
import sys
import shutil




try:
    #create empty workspace
    os.system("ocrd workspace init test_ws")
    #download test document image and models
    os.system("wget https://cloud.dfki.de/owncloud/index.php/s/SGJC8TrNWsH4gtC/download")
    os.rename("download", "test.tiff")
    #os.system("wget https://cloud.dfki.de/owncloud/index.php/s/3zKza5sRfQB3ygy/download")
    if not os.path.exists("test_ws/models"): 
    	os.mkdir("test_ws/models")
    #os.rename("download", "test_ws/models/latest_net_G.pth")
    os.system("wget https://cloud.dfki.de/owncloud/index.php/s/dgACCYzytxnb7Ey/download")
    os.rename("download", "test_ws/models/block_segmentation_weights.h5")
    #os.system("wget https://cloud.dfki.de/owncloud/index.php/s/E85PL48Cjs8ZkJL/download")
    #os.rename("download", "test_ws/models/structure_analysis.h5")
    #os.system("wget https://cloud.dfki.de/owncloud/index.php/s/skWCsWwq7ffM7aq/download")
    #os.rename("download", "test_ws/models/mapping_densenet.pickle")
    #add image to workspace
    os.system("ocrd workspace -d test_ws add --file-grp OCR-D-IMG --file-id TESTPAGE-0001 --mimetype image/png test.tiff")
    
    #modules (page level)
    os.system("ocrd-anybaseocr-binarize -m test_ws/mets.xml -I OCR-D-IMG -O OCR-D-PAGE-BIN")
    os.system("ocrd-anybaseocr-deskew -m test_ws/mets.xml -I OCR-D-PAGE-BIN -O OCR-D-PAGE-DESKEW")
    #os.system("ocrd-anybaseocr-dewarp -m test_ws/mets.xml -I OCR-D-PAGE-DESKEW -O OCR-D-PAGE-DEWARP")
    os.system("ocrd-anybaseocr-crop -m test_ws/mets.xml -I OCR-D-PAGE-DESKEW -O OCR-D-PAGE-CROP")
    os.system("ocrd-anybaseocr-tiseg -m test_ws/mets.xml -I OCR-D-PAGE-CROP -O OCR-D-PAGE-TISEG")
    os.system("ocrd-anybaseocr-block-segmentation -m test_ws/mets.xml -I OCR-D-PAGE-TISEG -O OCR-D-PAGE-BLOCK-SEG")
    #os.system("ocrd-anybaseocr-layout-analysis -m test_ws/mets.xml -I OCR-D-PAGE-BIN -O OCR-D-PAGE-STRUCT")

    #modules on region level
    #os.system("ocrd-anybaseocr-block-segmentation -m test_ws/mets.xml -I OCR-D-PAGE-BIN -O OCR-D-PAGE-BLOCK-SEG")
    #os.system("ocrd-anybaseocr-binarize -m test_ws/mets.xml -I OCR-D-PAGE-BLOCK-SEG -O OCR-D-PAGE-BIN-REG") 
    #os.system("ocrd-anybaseocr-deskew -m test_ws/mets.xml -I OCR-D-PAGE-BLOCK-SEG -O OCR-D-PAGE-DESKEW-REG")
    #os.system("ocrd-anybaseocr-dewarp -m test_ws/mets.xml -I OCR-D-PAGE-BLOCK-SEG -O OCR-D-PAGE-DEWARP-REG")  
    os.system("ocrd-anybaseocr-textline -m test_ws/mets.xml -I OCR-D-PAGE-BLOCK-SEG -O OCR-D-PAGE-TL-REG")

except Exception as e:
    print(e)
    #shutil.rmtree("test_ws")
    #os.system("rm test.tiff")
