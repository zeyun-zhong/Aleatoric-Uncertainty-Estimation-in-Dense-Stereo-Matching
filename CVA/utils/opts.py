import argparse

def parse_train_opts():
    parser = argparse.ArgumentParser(description='Training setup')
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--name', default=None, type=str, required=False, help='folder name')
    parser.add_argument('--cv_method', default='Census-BM', type=str, required=True, help='method to create cv')
    parser.add_argument('--dataset', default='K12', type=str, required=False, help='dataset used for training')
    parser.add_argument('--loss_type', default='Laplacian', type=str, required=True, help='loss type for training')
    parser.add_argument('--cluster', default=False, action='store_true')
    parser.add_argument('--no-cluster', dest='cluster', action='store_false')
    parser.add_argument('--lr', default=0.0001, type=float, required=False)
    parser.add_argument('--batch_size', default=128, type=int, required=False)
    parser.add_argument('--dense_layer_type', default='AP', type=str, required=False)
    parser.add_argument('--dense_filter_num', default=16, type=int, required=False)
    parser.add_argument('--nb_filter_num', default=32, type=int, required=False, help='filter nums in neighbourhood fusion')
    parser.add_argument('--depth_filter_num', default=32, type=int, required=False, help='filter nums in depth processing')
    parser.add_argument('--nb_filter_size', default=5, type=int, required=False, help='filter size in nb fusion part')

    # Geometry-aware model with mask predictions, coefficient for the binary cross-entropy term
    parser.add_argument('--geometry_loss_weight', default=1.0, type=float, required=False, help='coefficient for binary cross-entropy term')

    # GMM parameters
    parser.add_argument('--gmm_loss_weight', default=1.0, type=float, required=False, help='coefficient for mode shift error of GMM')
    parser.add_argument('--num_gaussian_comp', default=3, type=int, required=False, help='number of mixture components')

    # Weighted loss, weighted regression
    parser.add_argument('--using_weighted_loss', default=False, action='store_true')
    parser.add_argument('--not-using_weighted_loss', dest='using_weighted_loss', action='store_false')

    parser.add_argument('--using_lookup', default=False, action='store_true')
    parser.add_argument('--not-using_lookup', dest='using_lookup', action='store_false')

    args = parser.parse_args()
    return args