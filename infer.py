import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')
from utils import combine_csv_files_to_dataframe,create_directory, smooth_and_plot, smooth_and_compute_derivative
# CV pipes
from cv_dataloader import create_image_data_loader
from cv_model import create_model
from cv_train import save_cv_predictions
# Timeseries pipes
from model import TransformerModel_ver2
from dataloader import create_data_loader
from train import save_predictions

keep_cols = ['file_name',
       'binary_Некачественное ГДИС', 
       'binary_Влияние ствола скважины', 
       'binary_Радиальный режим',
       'binary_Линейный режим', 
       'binary_Билинейный режим', 
       'binary_Сферический режим', 
       'binary_Граница постоянного давления',
       'binary_Граница непроницаемый разлом',
       'regression_Влияние ствола скважины_details',
       'regression_Радиальный режим_details',
       'regression_Линейный режим_details',
       'regression_Билинейный режим_details',
       'regression_Сферический режим_details',
       'regression_Граница постоянного давления_details',
       'regression_Граница непроницаемый разлом_details']
columns_b=['Некачественное ГДИС',
       'Влияние ствола скважины', 'Радиальный режим', 'Линейный режим',
       'Билинейный режим', 'Сферический режим', 'Граница постоянного давления',
       'Граница непроницаемый разлом', ]
columns_r=['Влияние ствола скважины_details',
       'Радиальный режим_details', 'Линейный режим_details',
       'Билинейный режим_details', 'Сферический режим_details',
       'Граница постоянного давления_details',
       'Граница непроницаемый разлом_details']

columns_b_2 = ['binary_Некачественное ГДИС', 
       'binary_Влияние ствола скважины', 
       'binary_Радиальный режим',
       'binary_Линейный режим', 
       'binary_Билинейный режим', 
       'binary_Сферический режим', 
       'binary_Граница постоянного давления',
       'binary_Граница непроницаемый разлом',]
columns_r_2 =['regression_Влияние ствола скважины_details',
       'regression_Радиальный режим_details',
       'regression_Линейный режим_details',
       'regression_Билинейный режим_details',
       'regression_Сферический режим_details',
       'regression_Граница постоянного давления_details',
       'regression_Граница непроницаемый разлом_details']
def blend_them(dfs_b, dfs_r):
    # for df in dfs_b+dfs_r:
    #     df = df.sort_values('file_name')
    # for df in dfs_r:
    #     df = apply_trans_reg(df, path_to_scalers='E:/DS/data/siam/scalers/tg_scalers_r/tg_r_')
    res = dfs_b[0][['file_name']].copy()
    for col in columns_r_2:
        base = dfs_r[0][col].values/len(dfs_r)
        if len(dfs_r)>1:
            for df in dfs_r[1:]:
                base += df[col].values/len(dfs_r)
        res[col]=base
    for col in columns_b_2:
        col_ = col
        col =col+'_prob'
        base = dfs_b[0][col].values/len(dfs_b)
        if len(dfs_b)>1:
            for df in dfs_b[1:]:
                base += df[col].values/len(dfs_b)
        res[col_]=((base>0.5)*1)
    res = res[keep_cols]
    res.columns = ['file_name']+columns_b+columns_r
    return res
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple inference script')
    parser.add_argument('--path_to_csv_dir', type=str, required=True, help='Path to directory containing CSV files')
    return parser.parse_args()

def process_csv_directory(PATH_TO_DATA: str):
    """Process the CSV directory and return results.
    
    Args:
        path: Path to the directory containing CSV files
    """
    # Your processing logic here
    print('creating directories')
    create_directory(PATH_TO_DATA,'preproc_ts')
    create_directory(PATH_TO_DATA,'plots')
    create_directory(PATH_TO_DATA,'preds')
    print('combining csv files to dataframe')
    df = combine_csv_files_to_dataframe(PATH_TO_DATA)
    print('saving dataframe to parquet')
    df.to_parquet(Path(PATH_TO_DATA) / 'preproc_ts/source.parquet', index=False)
    print('generating features')
    test = df.copy()
    un_ids = test.file_name.unique()
    new_test = []
    for uuid in tqdm(un_ids, total=len(un_ids)):
        cc = test[test.file_name==uuid]
        cc = smooth_and_compute_derivative(cc)
        new_test.append(cc)
    new_test = pd.concat(new_test)
    print('log transforming features')
    df[df.columns[1:]] = np.log1p(df[df.columns[1:]].abs())
    new_test[new_test.columns[1:]] = np.log1p(new_test[new_test.columns[1:]].abs())
    print('saving features to parquet')
    df.to_parquet(Path(PATH_TO_DATA) / 'preproc_ts/feat3_l1p.parquet', index=False)
    new_test.to_parquet(Path(PATH_TO_DATA) / 'preproc_ts/feat10_l1p.parquet', index=False)

def gen_plots(PATH_TO_DATA: str):
    """Generate plots for the CSV directory.
    
    Args:
        PATH_TO_DATA: Path to the directory containing CSV files
    """
    print('reading parquet')
    test = pd.read_parquet(Path(PATH_TO_DATA) / 'preproc_ts/source.parquet')
    #new_train = []
    print('generating plots')
    un_ids = test.file_name.unique()
    errors = []
    for uuid in tqdm(un_ids, total = len(un_ids)):
        #file_path = os.path.join('E:/DS/data/siam/bi/test/', uuid)
        #if os.path.isfile(file_path):
            # Read the CSV file without a header and with tab as the separator
            #try:
                #df = pd.read_csv(file_path, sep='\t', header=None)
        
                # Ensure the file has exactly 3 columns (time, atm, der_atm)
        
                # Assign column names to the DataFrame
                #df.columns = ['time', 'atm', 'der_atm']
        
                # Add a new column for the file name
                #df['file_name'] = uuid
        
                # Reorder columns to match the desired output format
                #df = df[['file_name', 'time', 'atm', 'der_atm']]
        cc = test[test.file_name==uuid].copy()
        try:
            smooth_and_plot(cc, Path(PATH_TO_DATA) / 'plots/', pressure_type='origin', derivative_type='origin', save_plot=True)
            #new_train.append(cc)
        except:
            errors.append(uuid)
            # except:
            #     errors.append(uuid)
def load_models():
    print('loading models')
    swin_v6 =  create_model(
        model_type = "swin",
        #model_name='b7',
        n_binary_targets = 8,
        n_regression_targets = 7,
        dropout = 0.1,
        freeze_backbone = False,
        use_multi_layer_features = False,
        pretrained = True,
        weights_path='./weights/swin_v6_l1p.pth'
    )
    vit_v4 =  create_model(
        model_type = "vit",
        #model_name='b7',
        n_binary_targets = 8,
        n_regression_targets = 7,
        dropout = 0.1,
        freeze_backbone = False,
        use_multi_layer_features = False,
        pretrained = True,
        weights_path='./weights/vit_v4_l1p.pth'
    )
    conv_v5 =  create_model(
        model_type = "convnext",
        #model_name='b7',
        n_binary_targets = 8,
        n_regression_targets = 7,
        dropout = 0.1,
        freeze_backbone = False,
        use_multi_layer_features = False,
        pretrained = True,
        weights_path='./weights/conv_v5_l1p.pth'
    )
    model_ts10 = TransformerModel_ver2(
    input_dim=10,
    d_model=256,
    nhead=4,
    num_encoder_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_len = 1024
    )
    model_ts10.load_state_dict(torch.load("./weights/ts_10f_l1p_1.pth"))
    model_ts10.eval()

    model_ts3 = TransformerModel_ver2(
        input_dim=3,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len = 1024
    )
    model_ts3.load_state_dict(torch.load("./weights/ts_3f_l1p_1.pth"))
    model_ts3.eval()
    return model_ts3, model_ts10, conv_v5, vit_v4, swin_v6

def get_dataloaders(PATH_TO_DATA: str):
    """Get dataloaders for the inference process."""
    print('creating dataloaders')
    cv_loader  = create_image_data_loader(
        csv_path=None,
        images_dir=Path(PATH_TO_DATA) / 'plots/',
        batch_size = 64,
        shuffle = False,
        num_workers = 0
    )
    test_loader_10f  = create_data_loader(
        csv_path=None,
        parquet_path=Path(PATH_TO_DATA) / 'preproc_ts/feat10_l1p.parquet',
        batch_size=64,
        max_seq_len=1024,
        num_workers=0
    )
    test_loader_3f  = create_data_loader(
        csv_path=None,
        parquet_path=Path(PATH_TO_DATA) / 'preproc_ts/feat3_l1p.parquet',
        batch_size=64,
        max_seq_len=1024,
        num_workers=0
    )
    return cv_loader, test_loader_10f, test_loader_3f
def main():
    """Main function to orchestrate the inference process."""
    # Get command line arguments
    args = parse_arguments()
    PATH_TO_DATA=args.path_to_csv_dir
    process_csv_directory(PATH_TO_DATA)
    gen_plots(PATH_TO_DATA)
    model_ts3, model_ts10, conv_v5, vit_v4, swin_v6 = load_models()
    cv_loader, test_loader_10f, test_loader_3f = get_dataloaders(PATH_TO_DATA)
    print('saving cv predictions')
    save_cv_predictions(swin_v6,cv_loader,
                    Path(PATH_TO_DATA)/'preds/inf_CV_swin_v6_l1p.csv',
                    'cuda',
                    binary_threshold=0.5)
    save_cv_predictions(vit_v4,cv_loader,
                    Path(PATH_TO_DATA)/'preds/inf_CV_vit_v4_l1p.csv',
                    'cuda',
                    binary_threshold=0.5)
    save_cv_predictions(conv_v5,cv_loader,
                    Path(PATH_TO_DATA)/'preds/inf_CV_conv_v5_l1p.csv',
                    'cuda',
                    binary_threshold=0.5)
    save_predictions(model_ts3,test_loader_3f,
                Path(PATH_TO_DATA)/'preds/inf_l1p_ts3_v1.csv',
                'cuda',
                binary_threshold=0.5)
    save_predictions(model_ts10,test_loader_10f,
                    Path(PATH_TO_DATA)/'preds/inf_l1p_ts10_v1.csv',
                    'cuda',
                    binary_threshold=0.5)
    print('blending predictions')
    # TS_3 l1p v1 Результат: 0.6881 with weights
    ch8 =pd.read_csv(Path(PATH_TO_DATA)/'preds/inf_l1p_ts3_v1.csv').sort_values('file_name')
    ch8[columns_r_2] = np.expm1(ch8[columns_r_2])   
    # TS 10   l1p  Результат: 0.6866 with weights  inf_l1p_ts10_v1
    ch10 = pd.read_csv(Path(PATH_TO_DATA)/'preds/inf_l1p_ts10_v1.csv').sort_values('file_name')
    ch10[columns_r_2] = np.expm1(ch10[columns_r_2])
    # VS Swin l1p  Результат: 0.6189 with weights
    ch11 = pd.read_csv(Path(PATH_TO_DATA)/'preds/inf_CV_swin_v6_l1p.csv').sort_values('file_name')
    ch11[columns_r_2] = np.expm1(ch11[columns_r_2])
    # VS Conv l1p v1 Результат: 0.5541 with weights
    ch9 = pd.read_csv( Path(PATH_TO_DATA)/'preds/inf_CV_conv_v5_l1p.csv').sort_values('file_name')
    ch9[columns_r_2] = np.expm1(ch9[columns_r_2])
    # VS vit l1p v4 Результат: Результат: 0.517 with weights
    ch12 = pd.read_csv(Path(PATH_TO_DATA)/'preds/inf_CV_vit_v4_l1p.csv').sort_values('file_name')
    ch12[columns_r_2] = np.expm1(ch12[columns_r_2])
    #Результат: 0.7297
    res3_2 =  blend_them(dfs_b=[ch8,ch12, ch9,ch10,ch11], dfs_r=[ch8,ch10])
    create_directory(PATH_TO_DATA,'FINAL')
    res3_2.to_csv(Path(PATH_TO_DATA)/'FINAL/blend3_2.csv',index=False)
    print('FINAL path:', Path(PATH_TO_DATA)/'FINAL/blend3_2.csv')
if __name__ == "__main__":
    main() 