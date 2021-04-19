import argparse

def _parse_train_opts():
    parser = argparse.ArgumentParser(description='Training setup')
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--name', default=None, type=str, required=False, help='folder name')
    parser.add_argument('--cv_method', default='Census-BM', type=str, required=True, help='method to create cv')
    parser.add_argument('--dataset', default='K12', type=str, required=False, help='dataset used for training')
    parser.add_argument('--loss_type', default='Probabilistic', type=str, required=True, help='loss type for training')
    # parser.add_argument('--luis_cluster', default=False, action="store_true", help='local or luis cluster')
    parser.add_argument('--cluster', default=False, action='store_true')
    parser.add_argument('--no-cluster', dest='cluster', action='store_false')
    parser.add_argument('--lr', default=0.0001, type=float, required=False)
    parser.add_argument('--dense_layer_type', default='AP', type=str, required=False)
    parser.add_argument('--dense_filter_num', default=16, type=int, required=False)
    parser.add_argument('--nb_filter_num', default=32, type=int, required=False, help='filter nums in neighbourhood fusion')
    parser.add_argument('--depth_filter_num', default=32, type=int, required=False, help='filter nums in depth processing')
    parser.add_argument('--nb_filter_size', default=5, type=int, required=False, help='filter size in nb fusion part')

    # Geometry-aware, coefficient for binary cross-entropy term
    parser.add_argument('--eta', default=1.0, type=float, required=False, help='coefficient for binary cross-entropy term')

    # GMM parameters
    parser.add_argument('--gamma', default=1.0, type=float, required=False, help='coefficient for mode shift error of GMM')
    parser.add_argument('--K', default=3, type=int, required=False, help='number of mixture component')

    # Laplacian Uniform parameters
    parser.add_argument('--lu_out', default=3, type=int, required=False,
                        help='whether stds of Gaussian and Uniform should be the same')

    args = parser.parse_args()
    return args