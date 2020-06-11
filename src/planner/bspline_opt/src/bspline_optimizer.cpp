#include "bspline_opt/bspline_optimizer.h"
#include "bspline_opt/gradient_descent_optimizer.h"
// #include <nlopt.hpp>
// using namespace std;

namespace rebound_planner {

void BsplineOptimizer::setParam(ros::NodeHandle& nh) {
  nh.param("optimization/lambda_smooth", lambda1_, -1.0);
  nh.param("optimization/lambda_collision", lambda2_, -1.0);
  nh.param("optimization/lambda_feasibility", lambda3_, -1.0);
  nh.param("optimization/lambda_fitness", lambda4_, -1.0);

  nh.param("optimization/dist0", dist0_, -1.0);
  nh.param("optimization/max_vel", max_vel_, -1.0);
  nh.param("optimization/max_acc", max_acc_, -1.0);

  nh.param("optimization/order", order_, 3);
}

void BsplineOptimizer::setEnvironment(const SDFMap::Ptr& env) {
  this->sdf_map_ = env;
}

void BsplineOptimizer::setControlPoints(const Eigen::MatrixXd& points) {
  cps_.points.clear();
  cps_.points.reserve(points.rows());
  for ( int i=0; i<points.rows(); i++ )
  {
    cps_.points.push_back( points.row(i).transpose() );
  }
}

void BsplineOptimizer::setBsplineInterval(const double& ts) { bspline_interval_ = ts; }

std::vector<std::vector<Eigen::Vector3d>> BsplineOptimizer::initControlPoints(std::vector<Eigen::Vector3d> &init_points, bool flag_first_init /*= true*/)
{
  if ( flag_first_init )
  {
    
    cps_.clearance = dist0_;
    cps_.resize( init_points.size() );

    cps_.points = init_points;
  }

  /*** Segment the initial trajectory according to obstacles ***/
  constexpr int ENOUGH_INTERVAL = 2;
  double step_size = sdf_map_->getResolution() / ( (init_points[0] - init_points.back()).norm() / (init_points.size()-1) ) / 2;
  //cout << "step_size = " << step_size << endl;
  int in_id, out_id;
  vector<std::pair<int,int>> segment_ids;
  int same_occ_times = ENOUGH_INTERVAL + 1;
  bool occ, last_occ = false;
  bool flag_got_start = false, flag_got_end = false, flag_got_end_maybe = false;
  for ( int i=order_; i<=(int)init_points.size()-order_; ++i )
  {
    for ( double a=1.0; a>=0.0; a-=step_size )
    {
      occ = sdf_map_->getInflateOccupancy(a * init_points[i-1] + (1-a) * init_points[i]);

      // cout << "occ = " << occ << "  p = " << (a * init_points[i-1] + (1-a) * init_points[i]).transpose() << endl;
      // cout << "i=" << i <<endl;

      if ( occ && ! last_occ)
      {
        if ( same_occ_times > ENOUGH_INTERVAL || i == order_ )
        {
          in_id = i-1;
          flag_got_start = true;
        }
        same_occ_times = 0;
        flag_got_end_maybe = false;  // terminate in advance
      }
      else if( !occ && last_occ )
      {
        out_id = i;
        flag_got_end_maybe = true;
        same_occ_times = 0;
      }
      else
      {
        ++ same_occ_times; 
      }

      if ( flag_got_end_maybe && ( same_occ_times > ENOUGH_INTERVAL || ( i == (int)init_points.size()-order_ ) ) )
      {
        flag_got_end_maybe = false;
        flag_got_end = true;
      }
      
      last_occ = occ;
      
      if ( flag_got_start && flag_got_end )
      {
        flag_got_start = false;
        flag_got_end = false;
        segment_ids.push_back( std::pair<int,int>(in_id, out_id) );
      }
    }
  }


  /*** a star search ***/
  vector<vector<Eigen::Vector3d>> a_star_pathes;
  for ( size_t i=0; i<segment_ids.size(); ++i )
  {
    //cout << "in=" << in.transpose() << " out=" << out.transpose() << endl;
    Eigen::Vector3d in( init_points[segment_ids[i].first] ), out( init_points[segment_ids[i].second] );
    if ( a_star_->AstarSearch( /*(in-out).norm()/10+0.05*/0.1 , in, out) )
    {
      a_star_pathes.push_back( a_star_->getPath() );
    }
    else
    {
      ROS_ERROR("a star error, force return!");
      return a_star_pathes;
    }
  }

  // for ( int j=0; j<segment_ids.size(); ++j )
  // {
  //   cout << "------------ " << segment_ids[j].first << " " << segment_ids[j].second <<  endl;
  //   cout.precision(3);
  //   cout << "in=" << cps_[segment_ids[j].first].point.transpose() << " out=" << cps_[segment_ids[j].second].point.transpose() << endl;
  //   for ( int k=0; k<a_star_pathes[j].size(); ++k )
  //   {
  //     cout << "a_star_pathes[j][k]=" << a_star_pathes[j][k].transpose() << endl;
  //   }
  // }    

  /*** calculate bounds ***/
  int id_low_bound, id_up_bound;
  vector<std::pair<int,int>> bounds( segment_ids.size() );
  for (size_t i=0; i<segment_ids.size(); i++)
  {

    if ( i == 0 ) // first segment
    {
      id_low_bound = order_;
      if ( segment_ids.size() > 1 )
      {
        id_up_bound =  (int)(((segment_ids[0].second + segment_ids[1].first)-1.0f) / 2); // id_up_bound : -1.0f fix()
      }
      else
      {
        id_up_bound = init_points.size() - order_ - 1;
      }
    }
    else if ( i == segment_ids.size() - 1 ) // last segment, i != 0 here
    {
      id_low_bound = (int)(((segment_ids[i].first + segment_ids[i-1].second)+1.0f) / 2);   // id_low_bound : +1.0f ceil()
      id_up_bound = init_points.size() - order_ - 1;
    }
    else
    {
      id_low_bound = (int)(((segment_ids[i].first + segment_ids[i-1].second)+1.0f) / 2);  // id_low_bound : +1.0f ceil()
      id_up_bound = (int)(((segment_ids[i].second + segment_ids[i+1].first)-1.0f) / 2);  // id_up_bound : -1.0f fix()
    }
    
    bounds[i] = std::pair<int,int>(id_low_bound, id_up_bound);
  }

  // cout << "+++++++++" << endl;
  // for ( int j=0; j<bounds.size(); ++j )
  // {
  //   cout << bounds[j].first << "  " << bounds[j].second << endl;
  // }

  /*** Adjust segment length ***/
  vector<std::pair<int,int>> final_segment_ids( segment_ids.size() );
  constexpr double MINIMUM_PERCENT = 0.0;  // Each segment is guaranteed to have sufficient points to generate sufficient thrust
  int minimum_points = round(init_points.size() * MINIMUM_PERCENT), num_points;
  for (size_t i=0; i<segment_ids.size(); i++)
  {
    /*** Adjust segment length ***/
    num_points = segment_ids[i].second - segment_ids[i].first + 1;
    //cout << "i = " << i << " first = " << segment_ids[i].first << " second = " << segment_ids[i].second << endl;
    if ( num_points < minimum_points )
    {
      double add_points_each_side = (int)(((minimum_points - num_points)+1.0f) / 2);

      final_segment_ids[i].first = segment_ids[i].first - add_points_each_side >= bounds[i].first ?
        segment_ids[i].first - add_points_each_side :
        bounds[i].first;

      final_segment_ids[i].second = segment_ids[i].second + add_points_each_side <= bounds[i].second ?
        segment_ids[i].second + add_points_each_side :
        bounds[i].second;
    }
    else
    {
      final_segment_ids[i].first = segment_ids[i].first;
      final_segment_ids[i].second = segment_ids[i].second;
    }
    
    //cout << "final:" << "i = " << i << " first = " << final_segment_ids[i].first << " second = " << final_segment_ids[i].second << endl;
  }

  /*** Assign parameters to each segment ***/
  for (size_t i=0; i<segment_ids.size(); i++)
  {
    // step 1
    for ( int j=final_segment_ids[i].first; j <= final_segment_ids[i].second; ++j )
      cps_.flag_temp[j] = false;

    // step 2
    int got_intersection_id = -1;
    for ( int j=segment_ids[i].first+1; j<segment_ids[i].second; ++j )
    {
      Eigen::Vector3d ctrl_pts_law(cps_.points[j+1] - cps_.points[j-1]), intersection_point;
      int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id; // Let "Astar_id = id_of_the_most_far_away_Astar_point" will be better, but it needs more computation
      double val = (a_star_pathes[i][Astar_id] - cps_.points[j]).dot( ctrl_pts_law ), last_val = val;
      while ( Astar_id >=0 && Astar_id < (int)a_star_pathes[i].size() )
      {
        last_Astar_id = Astar_id;

        if ( val >= 0 )
          -- Astar_id;
        else
          ++ Astar_id;
        
        val = (a_star_pathes[i][Astar_id] - cps_.points[j]).dot( ctrl_pts_law );
        
        if ( val * last_val <= 0 && ( abs(val) > 0 || abs(last_val) > 0 ) ) // val = last_val = 0.0 is not allowed
        {
          intersection_point = 
            a_star_pathes[i][Astar_id] + 
            ( ( a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id] ) * 
              ( ctrl_pts_law.dot( cps_.points[j] - a_star_pathes[i][Astar_id] ) / ctrl_pts_law.dot( a_star_pathes[i][Astar_id] -  a_star_pathes[i][last_Astar_id] ) ) // = t
            );

          //cout << "i=" << i << " j=" << j << " Astar_id=" << Astar_id << " last_Astar_id=" << last_Astar_id << " intersection_point = " << intersection_point.transpose() << endl;

          got_intersection_id = j;
          break;
        }
      }

      if ( got_intersection_id >= 0 )
      {
        cps_.flag_temp[j] = true;
        double length = (intersection_point - cps_.points[j]).norm();
        if ( length > 1e-5 )
        {
          for ( double a=length; a>=0.0; a-=sdf_map_->getResolution() )
          {
            occ =  sdf_map_->getInflateOccupancy((a/length)*intersection_point + (1-a/length)*cps_.points[j]);
      
            if ( occ || a < sdf_map_->getResolution() )
            {
              if ( occ )
                a+=sdf_map_->getResolution();
              cps_.base_point[j].push_back( (a/length)*intersection_point + (1-a/length)*cps_.points[j] );
              cps_.direction[j].push_back( (intersection_point - cps_.points[j]).normalized() );
              break;
            }
          }
        }
      }
    }

    /* Corner case: the segment length is too short. Here the control points may outside the A* path, leading to opposite gradient direction. So I have to take special care of it */
    if ( segment_ids[i].second - segment_ids[i].first == 1 ) 
    {
      Eigen::Vector3d ctrl_pts_law(cps_.points[segment_ids[i].second] - cps_.points[segment_ids[i].first]), intersection_point;
      Eigen::Vector3d middle_point = (cps_.points[segment_ids[i].second] + cps_.points[segment_ids[i].first]) / 2;
      int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id; // Let "Astar_id = id_of_the_most_far_away_Astar_point" will be better, but it needs more computation
      double val = (a_star_pathes[i][Astar_id] - middle_point).dot( ctrl_pts_law ), last_val = val;
      while ( Astar_id >=0 && Astar_id < (int)a_star_pathes[i].size() )
      {
        last_Astar_id = Astar_id;

        if ( val >= 0 )
          -- Astar_id;
        else
          ++ Astar_id;
        
        val = (a_star_pathes[i][Astar_id] - middle_point).dot( ctrl_pts_law );
        
        if ( val * last_val <= 0 && ( abs(val) > 0 || abs(last_val) > 0 ) ) // val = last_val = 0.0 is not allowed
        {
          intersection_point = 
            a_star_pathes[i][Astar_id] + 
            ( ( a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id] ) * 
              ( ctrl_pts_law.dot( middle_point - a_star_pathes[i][Astar_id] ) / ctrl_pts_law.dot( a_star_pathes[i][Astar_id] -  a_star_pathes[i][last_Astar_id] ) ) // = t
            );

          cps_.flag_temp[segment_ids[i].first] = true;
          cps_.base_point[segment_ids[i].first].push_back( cps_.points[segment_ids[i].first] );
          cps_.direction[segment_ids[i].first].push_back( (intersection_point - middle_point).normalized() );

          got_intersection_id = segment_ids[i].first;
          break;
        }
      }
    }

    //step 3
    if ( got_intersection_id >= 0 )
    {
      for ( int j=got_intersection_id + 1; j <= final_segment_ids[i].second; ++j )
        if ( ! cps_.flag_temp[j] )
        {
          cps_.base_point[j].push_back( cps_.base_point[j-1].back() );
          cps_.direction[j].push_back( cps_.direction[j-1].back() );
        }

      for ( int j=got_intersection_id - 1; j >= final_segment_ids[i].first; --j )
        if ( ! cps_.flag_temp[j] )
        {
          cps_.base_point[j].push_back( cps_.base_point[j+1].back() );
          cps_.direction[j].push_back( cps_.direction[j+1].back() );
        }
    }
    else
    {
      ROS_ERROR("Failed to generate direction! segment_id=%d", i);
      
      // cout << "↓↓↓↓↓↓↓↓↓↓↓↓ " << final_segment_ids[i].first << " " << final_segment_ids[i].second << endl;
      // cout.precision(3);
      // cout << "in=" << cps_[final_segment_ids[i].first].point.transpose() << " out=" << cps_[final_segment_ids[i].second].point.transpose() << endl;
      // for ( size_t k=0; k<a_star_pathes[i].size(); ++k )
      // {
      //   cout << "a_star_pathes[i][k]=" << a_star_pathes[i][k].transpose() << endl;
      // }
      // cout << "↑↑↑↑↑↑↑↑↑↑↑↑" << endl;
    }

  }

  return a_star_pathes;
}


double BsplineOptimizer:: costFunctionRebound(const Eigen::VectorXd& x, Eigen::VectorXd& grad, void* func_data)
{
  BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(func_data);

  double cost;
  opt->combineCostRebound(x, grad, cost);

  /* save the min cost result */
  opt->min_cost_ = cost;
  opt->best_variable_ = x;

  // early termination
  if ( opt->flag_continue_to_optimize_ )
  {
    cost = std::numeric_limits<double>::max();
    // for ( size_t i=0; i<grad.size(); i++)
    // {
    //   grad[i] = std::numeric_limits<double>::max();
    // }
  }

  // cout << "opt->flag_continue_to_optimize_=" << opt->flag_continue_to_optimize_ << endl;
  // cout << "cost=" << cost <<endl;

  opt->iter_num_ += 1;
  return cost;
}

double BsplineOptimizer::costFunctionRefine(const Eigen::VectorXd& x, Eigen::VectorXd& grad, void* func_data)
{
  BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(func_data);

  double cost;
  opt->combineCostRefine(x, grad, cost);

  /* save the min cost result */
  opt->min_cost_ = cost;
  opt->best_variable_ = x;

  opt->iter_num_ += 1;
  return cost;
}

void BsplineOptimizer::calcDistanceCostRebound(const vector<Eigen::Vector3d>& q, double& cost,
                                        vector<Eigen::Vector3d>& gradient, int iter_num, double smoothness_cost)
{
  //time_satrt = ros::Time::now();

  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

  //ROS_WARN("iter_num=%d", iter_num);

  double dist;
  Eigen::Vector3d dist_grad;
  int end_idx = q.size() - order_;

  flag_continue_to_optimize_ = false;
  if ( iter_num > 3 && smoothness_cost / ( cps_.size - 2*order_) < 0.1 ) // 0.1 is an experimental value that indicates the trajectory is smooth enough, leftover shit!!!
  {
    flag_continue_to_optimize_ = check_collision_and_rebound();
  }


  // cout << "iter_num = " << iter_num << endl;
  // for ( int i=0; i<=cps_.size()-1; ++i )
  // {
  //   cout << "--------------" <<endl;
  //   cout.precision(3);
  //   cout << "i=" << i << " point = " << cps_[i].point.transpose() << endl;
  //   for ( int j=0; j<cps_[i].direction.size(); ++j )
  //   {
  //     cout.precision(3);
  //     cout << "dir = " << cps_[i].direction[j].transpose() << " colli = " << cps_[i].base_point[j].transpose() << endl;
  //   }
  // }
  // cout <<endl;

  /*** calculate distance cost and gradient ***/
  for ( auto i=order_; i<end_idx; ++i )
  {
    for ( size_t j=0; j<cps_.direction[i].size(); ++j )
    {
      dist = (cps_.points[i] - cps_.base_point[i][j]).dot(cps_.direction[i][j]);
      dist_grad = cps_.direction[i][j];
      if (dist < cps_.clearance)
      {
        if ( dist < -cps_.clearance )
        {
          // linear if the distance is too far.
          // cost += pow(dist - cps_[i].clearance, 2) - pow(dist + cps_[i].clearance, 2);
          cost += -4 * cps_.clearance * dist;  // == pow(dist - cps_[i].clearance, 2) - pow(dist + cps_[i].clearance, 2)
          gradient[i] += -4 * cps_.clearance * dist_grad;
          //cout << "run to here! i=" << i << " dist=" << dist << endl;
        }
        else
        {
          cost += pow(dist - cps_.clearance, 2);
          gradient[i] += 2.0 * (dist - cps_.clearance) * dist_grad;
        }
        
        // if ( iter_num <= 2 )
        // {
        //   cout << "[new xxx] iter_num=" << iter_num << " i=" << i << " cps_[i].direction.size()=" << cps_[i].direction.size() << " cost=" << cost << " gradient[i]" << gradient[i].transpose() << endl; 
        // }
      }
    }
  }

  // time_end = ros::Time::now();
  // cout << "time=" << (time_end - time_satrt).toSec()*1000000 << endl;

}

void BsplineOptimizer::calcFitnessCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient)
{
  //time_satrt = ros::Time::now();

  cost = 0.0;
  std::fill(gradient.begin(), gradient.end(), Eigen::Vector3d(0, 0, 0));

  int end_idx = q.size() - order_;

  // // def: f = x^2
  // for ( auto i=order_-1; i<end_idx+1; ++i )
  // {
  //   Eigen::Vector3d temp = (q[i-1]+4*q[i]+q[i+1])/6.0 - ref_pts_[i-1];
  //   cost += temp.squaredNorm();

  //   gradient[i-1] +=   temp/3.0;
  //   gradient[i]   += 4*temp/3.0;
  //   gradient[i+1] +=   temp/3.0;
  // }

  // def: f = |x*v|^2/a^2 + |x×v|^2/b^2
  double a2 = 25, b2 = 1;
  for ( auto i=order_-1; i<end_idx+1; ++i )
  {
    Eigen::Vector3d x = (q[i-1]+4*q[i]+q[i+1])/6.0 - ref_pts_[i-1];
    Eigen::Vector3d v = (ref_pts_[i]-ref_pts_[i-2]).normalized();

    double xdotv = x.dot(v);
    Eigen::Vector3d xcrossv = x.cross(v);

    double f = pow((xdotv),2)/a2 + pow(xcrossv.norm(),2)/b2;
    cost += f;

    Eigen::Matrix3d m;
    m << 0,-v(2),v(1), v(2),0,-v(0), -v(1),v(0),0;
    Eigen::Vector3d df_dx = 2*xdotv/a2*v + 2/b2*m*xcrossv;

    gradient[i-1] += df_dx/6;
    gradient[i] += 4*df_dx/6;
    gradient[i+1] += df_dx/6;
  }

}


void BsplineOptimizer::calcSmoothnessCost(const vector<Eigen::Vector3d>& q, double& cost,
                                          vector<Eigen::Vector3d>& gradient, bool falg_use_jerk/* = true*/) {
  
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient.begin(), gradient.end(), zero);

  if ( falg_use_jerk )
  {
    Eigen::Vector3d jerk, temp_j;

    // for (int i = 0; i < q.size(); i++)
    //   cout << "i=" << i << "@" << q[i].transpose() << endl;

    for (int i = 0; i < q.size() - 3; i++) {
      /* evaluate jerk */
      jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i];
      cost += jerk.squaredNorm();
      temp_j = 2.0 * jerk;
      /* jerk gradient */
      gradient[i + 0] += -temp_j;
      gradient[i + 1] += 3.0 * temp_j;
      gradient[i + 2] += -3.0 * temp_j;
      gradient[i + 3] += temp_j;

      // cout << "i=" << i << " jerk^2=" << jerk.squaredNorm()*1000 << endl;
    }

    // cout << endl;
  }
  else
  {    
    Eigen::Vector3d acc, temp_acc;

    for (int i = 0; i < q.size() - 2; i++) {
      /* evaluate jerk */
      acc = q[i + 2] - 2 * q[i + 1] + q[i];
      cost += acc.squaredNorm();
      temp_acc = 2.0 * acc;
      /* jerk gradient */
      gradient[i + 0] += temp_acc;
      gradient[i + 1] += -2.0 * temp_acc;
      gradient[i + 2] += temp_acc;
    }
  }
  
}

void BsplineOptimizer::calcFeasibilityCost(const vector<Eigen::Vector3d>& q, double& cost,
                                           vector<Eigen::Vector3d>& gradient) {
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient.begin(), gradient.end(), zero);

  /* abbreviation */
  double ts, vm2, am2, ts_inv2, ts_inv4;
  vm2 = max_vel_ * max_vel_;
  am2 = max_acc_ * max_acc_;

  ts      = bspline_interval_;
  ts_inv2 = 1 / ts / ts;
  ts_inv4 = ts_inv2 * ts_inv2;

  /* velocity feasibility */
  for (int i = 0; i < q.size() - 1; i++) {
    Eigen::Vector3d vi = (q[i + 1] - q[i])/ts;

    //cout << "temp_v * vi=" ;
    for (int j = 0; j < 3; j++) {
      if ( vi(j) > max_vel_ )
      {
        // cout << "fuck VEL" << endl;
        // cout << vi(j) << endl;
        cost += pow( vi(j)-max_vel_, 2 );

        gradient[i+0](j) += -2*(vi(j)-max_vel_)/ts;
        gradient[i+1](j) += 2*(vi(j)-max_vel_)/ts;
      }
      else if ( vi(j) < -max_vel_ )
      {
        cost += pow( vi(j)+max_vel_, 2 );

        gradient[i+0](j) += -2*(vi(j)+max_vel_)/ts;
        gradient[i+1](j) += 2*(vi(j)+max_vel_)/ts;
      }
      else
      {
        /* code */
      }
      //cout << 4.0 * vd * ts_inv2 * vi(j) << " ";
    }
    //cout << endl;
  }

  /* acceleration feasibility */
  for (int i = 0; i < q.size() - 2; i++) {
    Eigen::Vector3d ai = (q[i + 2] - 2 * q[i + 1] + q[i])*ts_inv2;

    //cout << "temp_a * ai=" ;
    for (int j = 0; j < 3; j++) 
    {
      if ( ai(j) > max_acc_ )
      {
        // cout << "fuck ACC" << endl;
        // cout << ai(j) << endl;
        cost += pow( ai(j)-max_acc_, 2 );

        gradient[i + 0](j) += 2*(ai(j)-max_acc_)*ts_inv2;
        gradient[i + 1](j) += -4*(ai(j)-max_acc_)*ts_inv2;
        gradient[i + 2](j) += 2*(ai(j)-max_acc_)*ts_inv2;
      }
      else if ( ai(j) < -max_acc_ )
      {
        cost += pow( ai(j)+max_acc_, 2 );

        gradient[i + 0](j) += 2*(ai(j)+max_acc_)*ts_inv2;
        gradient[i + 1](j) += -4*(ai(j)+max_acc_)*ts_inv2;
        gradient[i + 2](j) += 2*(ai(j)+max_acc_)*ts_inv2;
      }
      else
      {
        /* code */
      }
    }
    //cout << endl;
  }

}



bool BsplineOptimizer::check_collision_and_rebound(void)
{

  int end_idx = cps_.size - order_;

  /*** Check and segment the initial trajectory according to obstacles ***/
  int in_id, out_id;
  vector<std::pair<int,int>> segment_ids;
  bool flag_new_obs_valid = false;
  for ( int i=order_-1; i<=end_idx; ++i )
  {

    bool occ = sdf_map_->getInflateOccupancy(cps_.points[i]);

    /*** check if the new collision will be valid ***/
    if ( occ )
    {
      for ( size_t k=0; k<cps_.direction[i].size(); ++k )
      {
        cout.precision(2);
        //cout << "Test_02" << " i=" << i << " k=" << k << " direction[k]=" << cps_[i].direction[k].transpose() << " base_point[k]=" << cps_[i].base_point[k].transpose() << " point=" << cps_[i].point.transpose() << " dot=" << ( cps_[i].point - cps_[i].base_point[k] ).dot(cps_[i].direction[k]) << endl;
        //if ( dir.dot(cps_[j].direction[k]) > 1e-5 ) // the angle of two directions is smaller than 90 degree. 
        if ( ( cps_.points[i] - cps_.base_point[i][k] ).dot(cps_.direction[i][k]) < 1 * sdf_map_->getResolution() ) // current point is outside any of the collision_points. 
        {
          occ = false;
          //cout << "Test_00" << " flag_new_obs=" << flag_new_obs << " j=" << j << " k=" << k << " dir=" << dir.transpose() << " cps_[j].direction[k]=" << cps_[j].direction[k].transpose() << " dot=" << ( cps_[j].point - cps_[j].base_point[k] ).dot(cps_[j].direction[k]) << endl;
          break;
        }
      }
      //cout << "flag_new_obs_valid = " << flag_new_obs_valid << " iter = " << iter_num << endl;
    }

    if ( occ )
    {
      flag_new_obs_valid = true;
      // cout << "hit new obs, cp_id = " << i << " iter=" << iter_num_ << endl;

      int j;
      for ( j=i-1; j>=0; --j )
      {
        occ = sdf_map_->getInflateOccupancy(cps_.points[j]);
        if ( !occ )
        {
          in_id = j;
          break;
        }
      }
      if ( j < 0 ) // fail to get the obs free point
      {
        ROS_ERROR( "ERROR! the drone is in obstacle. This should not happen." );
        in_id = 0;
      }

      for ( j=i+1; j<cps_.size; ++j )
      {
        occ = sdf_map_->getInflateOccupancy(cps_.points[j]);
        if ( !occ )
        {
          out_id = j;
          break;
        }
      }
      if ( j >= cps_.size ) // fail to get the obs free point
      {
        ROS_WARN( "WARN! terminal point of the current trajectory is in obstacle, skip this planning." );
        return 0;
      }

      i = j+1;

      segment_ids.push_back( std::pair<int,int>(in_id, out_id) );
    }
  }


  if ( flag_new_obs_valid )
  {
    vector<vector<Eigen::Vector3d>> a_star_pathes;
    for ( size_t i=0; i<segment_ids.size(); ++i )
    {
      /*** a star search ***/
      Eigen::Vector3d in( cps_.points[segment_ids[i].first] ), out( cps_.points[segment_ids[i].second] );
      if ( a_star_->AstarSearch( /*(in-out).norm()/10+0.05*/0.1, in, out) )
      {
        a_star_pathes.push_back( a_star_->getPath() );
      }
      else
      {
        ROS_ERROR("a star error");
        segment_ids.erase( segment_ids.begin() + i );
        i--;
      }

    }

    // if (flag_record_intermediate_state_)
    // {
    //   for ( auto pts : a_star_pathes )
    //   {
    //     a_star_pathes_log_.push_back(pts);
    //   }
    // }


    // for ( int j=0; j<segment_ids.size(); ++j )
    // {
    //   cout << "------------" << endl << segment_ids[j].first << " " << segment_ids[j].second << endl;
    //   for ( int k=0; k<a_star_pathes[j].size(); ++k )
    //   {
    //     cout << "a_star_pathes[j][k]=" << a_star_pathes[j][k].transpose() << endl;
    //   }
    // }

    /*** Assign parameters to each segment ***/
    for (size_t i=0; i<segment_ids.size(); ++i)
    {
      // step 1
      for ( int j=segment_ids[i].first; j <= segment_ids[i].second; ++j )
        cps_.flag_temp[j] = false;

      // for ( auto x : segment_ids )
      // {
      //   cout << "first=" << x.first << " second=" << x.second << endl;
      // }
      // step 2
      int got_intersection_id = -1;
      for ( int j=segment_ids[i].first+1; j<segment_ids[i].second; ++j )
      {
        Eigen::Vector3d ctrl_pts_law(cps_.points[j+1] - cps_.points[j-1]), intersection_point;
        int Astar_id = a_star_pathes[i].size() / 2, last_Astar_id; // Let "Astar_id = id_of_the_most_far_away_Astar_point" will be better, but it needs more computation
        double val = (a_star_pathes[i][Astar_id] - cps_.points[j]).dot( ctrl_pts_law ), last_val = val;
        while ( Astar_id >=0 && Astar_id < (int)a_star_pathes[i].size() )
        {
          last_Astar_id = Astar_id;

          if ( val >= 0 )
            -- Astar_id;
          else
            ++ Astar_id;
          
          val = (a_star_pathes[i][Astar_id] - cps_.points[j]).dot( ctrl_pts_law );
          
          if ( val * last_val <= 0 && ( abs(val) > 0 || abs(last_val) > 0 ) ) // val = last_val = 0.0 is not allowed
          {
            intersection_point = 
              a_star_pathes[i][Astar_id] + 
              ( ( a_star_pathes[i][Astar_id] - a_star_pathes[i][last_Astar_id] ) * 
                ( ctrl_pts_law.dot( cps_.points[j] - a_star_pathes[i][Astar_id] ) / ctrl_pts_law.dot( a_star_pathes[i][Astar_id] -  a_star_pathes[i][last_Astar_id] ) ) // = t
              );

            //cout << "i=" << i << " j=" << j << " Astar_id=" << Astar_id << " last_Astar_id=" << last_Astar_id << " intersection_point = " << intersection_point.transpose() << endl;

            got_intersection_id = j;
            break;
          }
        }

        if ( got_intersection_id >= 0 )
        {
          cps_.flag_temp[j] = true;
          double length = (intersection_point - cps_.points[j]).norm();
          if ( length > 1e-5 )
          {
            for ( double a=length; a>=0.0; a-=sdf_map_->getResolution() )
            {
              bool occ = sdf_map_->getInflateOccupancy((a/length)*intersection_point + (1-a/length)*cps_.points[j]);
        
              if ( occ || a < sdf_map_->getResolution() )
              {
                if ( occ )
                  a+=sdf_map_->getResolution();
                cps_.base_point[j].push_back( (a/length)*intersection_point + (1-a/length)*cps_.points[j] );
                cps_.direction[j].push_back( (intersection_point - cps_.points[j]).normalized() );
                break;
              }
            }
          }
          else
          {
            got_intersection_id = -1;
          }
        }
      }

      //step 3
      if ( got_intersection_id >= 0 )
      {
        for ( int j=got_intersection_id + 1; j <= segment_ids[i].second; ++j )
          if ( ! cps_.flag_temp[j] )
          {
            cps_.base_point[j].push_back( cps_.base_point[j-1].back() );
            cps_.direction[j].push_back( cps_.direction[j-1].back() );
          }

        for ( int j=got_intersection_id - 1; j >= segment_ids[i].first; --j )
          if ( ! cps_.flag_temp[j] )
          {
            cps_.base_point[j].push_back( cps_.base_point[j+1].back() );
            cps_.direction[j].push_back( cps_.direction[j+1].back() );
          }
      }
      else
        ROS_WARN("Failed to generate direction!");
    }

    return true;
  }

  return false;
}

bool BsplineOptimizer::BsplineOptimizeTrajRebound(const Eigen::MatrixXd init_points, Eigen::MatrixXd& optimal_points, double ts)
{
  //setControlPoints(init_points);
  setBsplineInterval(ts);
  //setTerminateCond(max_num_id, max_time_id);

  bool flag_success =  rebound_optimize();

  optimal_points.resize(cps_.size,3);
  for ( size_t i=0; i<cps_.size; i++ )
  {
    optimal_points.row(i) = cps_.points[i];
  }

  return flag_success;
}


bool BsplineOptimizer::BsplineOptimizeTrajRefine(const Eigen::MatrixXd& init_points, const double ts, Eigen::MatrixXd& optimal_points)
{

  setControlPoints(init_points);
  setBsplineInterval(ts);
  //setTerminateCond(max_num_id, max_time_id);
  
  bool flag_success =  refine_optimize();

  optimal_points.resize(cps_.points.size(),3);
  for ( size_t i=0; i<cps_.points.size(); i++ )
  {
    optimal_points.row(i) = cps_.points[i].transpose();
  }

  return flag_success;
}


bool BsplineOptimizer::rebound_optimize()
{
  /* ---------- initialize solver ---------- */
  iter_num_ = 0;
  int start_id = order_;
  int end_id = this->cps_.size - order_;
  variable_num_ = 3 * (end_id - start_id);

  // cout << "variable_num_=" << variable_num_ << endl;

  GradientDescentOptimizer opt(variable_num_, BsplineOptimizer::costFunctionRebound, this);

  opt.set_maxeval(200);
  opt.set_min_grad(1e-2);

  /* ---------- init variables ---------- */
  double final_cost;

  ros::Time t0 = ros::Time::now(), t1, t2;
  int restart_nums = 0, rebound_times = 0;;
  bool flag_rebound, flag_occ, success;
  double original_lambda2 = lambda2_;
  constexpr int MAX_RESART_NUMS_SET = 3;
  do
  {
    /* ---------- prepare ---------- */
    min_cost_ = std::numeric_limits<double>::max();
    iter_num_ = 0;
    flag_rebound = false;
    flag_occ = false;
    success = false;

    Eigen::VectorXd q(variable_num_);
    for (size_t i = start_id; i < end_id; ++i)
    {
      for (int j = 0; j < 3; j++)
        q(3 * (i - start_id) + j) = cps_.points[i](j);
    }

    /* ---------- optimize ---------- */
    t1 = ros::Time::now();
    auto result = opt.optimize(q, final_cost);
    t2 = ros::Time::now();
    double time_ms = (t2-t1).toSec()*1000;
    double total_time_ms = (t2-t0).toSec()*1000;

    /* ---------- success temporary, check collision again ---------- */
    if ( result == GradientDescentOptimizer::FIND_MIN ||  result == GradientDescentOptimizer::REACH_MAX_ITERATION )
    {
      flag_rebound = false;

      for (size_t i = start_id; i < end_id; ++i)
      {
        for (int j = 0; j < 3; j++)
          cps_.points[i](j) = best_variable_[3 * (i - start_id) + j];
      }

      // check collision
      Eigen::MatrixXd control_points(cps_.size, 3);
      for ( int i=0; i<cps_.size; ++i )
      {
        control_points.row(i) = cps_.points[i].transpose();
      }
      NonUniformBspline traj =  NonUniformBspline(control_points, 3, bspline_interval_);
      double tm, tmp;
      traj.getTimeSpan(tm, tmp);
      constexpr double t_step = 0.02;
      for ( double t = tm; t<tmp; t+=t_step )
      {
        flag_occ = sdf_map_->getInflateOccupancy( traj.evaluateDeBoor(t) );
        if ( flag_occ )
        {
          //cout << "hit_obs, t=" << t << " P=" << traj.evaluateDeBoor(t).transpose() << endl;

          if ( t <= bspline_interval_ ) // First 3 control points in obstacles!
          {
            cout << cps_.points[1].transpose() << "\n"  << cps_.points[2].transpose() << "\n"  << cps_.points[3].transpose() << "\n" << cps_.points[4].transpose() << endl;
            ROS_ERROR("First 3 control points in obstacles! return false, t=%f",t);
            return false;
          }
          else if ( t > tmp-bspline_interval_ ) // Last 3 control points in obstacles!
          {
            cout << "P=" << traj.evaluateDeBoor(t).transpose() << endl;
            ROS_ERROR("Last 3 control points in obstacles! return false, t=%f",t);
            return false;
          }

          break;
        }
      }

      // cout << "first_step=" << endl;
      // cout << "tm=" << tm << " tmp=" << tmp << endl;
      // cout << "cps_.size()=" << cps_.size << endl;
      // for ( double t = tm; t<tmp; t+=t_step )
      // {
      //   cout << traj.evaluateDeBoor(t).transpose() << endl;
      // }

      if ( !flag_occ )
      {
        printf("\033[32miter(+1)=%d,time(ms)=%5.3f,total_t(ms)=%5.3f,cost=%5.3f\n\033[0m", iter_num_, time_ms, total_time_ms, final_cost);
        success = true;
      }
      else // restart
      {
        restart_nums++;
        vector<Eigen::Vector3d> control_points(cps_.size);
        for ( size_t i=0; i<cps_.size; i++ )
        {
          control_points[i] = cps_.points[i];
        }
        initControlPoints(control_points, false);
        lambda2_ *= 2;
                            

        printf("\033[32miter(+1)=%d,time(ms)=%5.3f,keep optimizing\n\033[0m", iter_num_, time_ms);
      }

    }
    else if ( result == GradientDescentOptimizer::RETURN_BY_ORDER )
    {
      flag_rebound = true;
      rebound_times ++;
      cout << "iter=" << iter_num_ << ",time(ms)=" << time_ms << ",rebound." << endl;
    }

  } while ( (flag_occ && restart_nums < MAX_RESART_NUMS_SET) || 
            (flag_rebound && rebound_times <= 100) 
          );

  lambda2_ = original_lambda2;

  // if ( restart_nums < max_restart_nums_set || rebound_times > 100 ) return true; // rebound_times > 100? why???
  return success;
}

// bool BsplineOptimizer::rebound_optimize_nlopt()
// {
//   /* ---------- initialize solver ---------- */
//   iter_num_ = 0;
//   int start_id = order_;
//   int end_id = this->cps_.size - order_;
//   variable_num_ = 3 * (end_id - start_id);

//   // cout << "variable_num_" << variable_num_ << endl;

//   nlopt::opt opt(nlopt::algorithm(11 /*LBFGS*/), variable_num_);


//   opt.set_min_objective(BsplineOptimizer::costFunctionRebound, this);

//   opt.set_maxeval(50);
//   opt.set_xtol_rel(1e-5);
//   opt.set_maxtime(time_limit);

//   /* ---------- init variables ---------- */
//   vector<double> q(variable_num_);
//   double final_cost;
//   // for (size_t i = start_id; i < end_id; ++i)
//   // {
//   //   for (int j = 0; j < 3; j++)
//   //     q[3 * (i - start_id) + j] = cps_[i].point(j);
//   // }

//   clock_t t0 = clock(), t1, t2;
//   int restart_nums = 0, rebound_times = 0;;
//   bool flag_occ = false;
//   bool success = false;
//   bool flag_nlopt_error_and_totally_fail = false;
//   double original_lambda2 = lambda2_;
//   constexpr int max_restart_nums_set = 3;
//   do
//   {
//     min_cost_ = std::numeric_limits<double>::max();
//     flag_continue_to_optimize_ = false;
//     iter_num_ = 0;
//     try
//     {
//       /* ---------- optimization ---------- */
//       // cout << "[Optimization]: begin-------------" << endl;
//       cout << fixed << setprecision(7);
//       t1 = clock();

//       for (size_t i = start_id; i < end_id; ++i)
//       {
//         for (int j = 0; j < 3; j++)
//           q[3 * (i - start_id) + j] = cps_.points[i](j);
//       }

//       /*nlopt::result result = */opt.optimize(q, final_cost);

//       t2 = clock();
//       /* ---------- get results ---------- */
//       double time_ms = (double)(t2-t1)/CLOCKS_PER_SEC*1000;
//       double total_time_ms = (double)(t2-t0)/CLOCKS_PER_SEC*1000;

//       for (size_t i = start_id; i < end_id; ++i)
//       {
//         for (int j = 0; j < 3; j++)
//           cps_.points[i](j) = best_variable_[3 * (i - start_id) + j];
//       }

//       // check collision
//       for ( int i=0; i<cps_.size; ++i )
//         control_points_.row(i) = cps_.points[i].transpose();
//       NonUniformBspline traj =  NonUniformBspline(control_points_, 3, bspline_interval_);
//       double tm, tmp;
//       traj.getTimeSpan(tm, tmp);
//       constexpr double t_step = 0.02;
//       for ( double t = tm; t<tmp; t+=t_step )
//       {
//         flag_occ = sdf_map_->getInflateOccupancy( traj.evaluateDeBoor(t) );
//         if ( flag_occ )
//         {
//           //cout << "hit_obs, t=" << t << " P=" << traj.evaluateDeBoor(t).transpose() << endl;

//           if ( t <= bspline_interval_ ) // First 3 control points in obstacles!
//           {
//             cout << cps_.points[1].transpose() << "\n"  << cps_.points[2].transpose() << "\n"  << cps_.points[3].transpose() << "\n" << cps_.points[4].transpose() << endl;
//             ROS_ERROR("First 3 control points in obstacles! return false, t=%f",t);
//             return false;
//           }
//           else if ( t > tmp-bspline_interval_ ) // First 3 control points in obstacles!
//           {
//             cout << "P=" << traj.evaluateDeBoor(t).transpose() << endl;
//             ROS_ERROR("Last 3 control points in obstacles! return false, t=%f",t);
//             return false;
//           }

//           goto hit_obs;
//         }
//       }
      
//       success = true;

//       hit_obs:; 

//       if ( !flag_occ )
//       {
//         printf("\033[32miter(+1)=%d,time=%5.3f,total_t=%5.3f,cost=%5.3f\n\033[0m", iter_num_, time_ms, total_time_ms, final_cost);
//       }
//       else
//       {
//         restart_nums++;
//         vector<Eigen::Vector3d> control_points(cps_.size);
//         for ( size_t i=0; i<cps_.size; i++ )
//         {
//           control_points[i] = cps_.points[i];
//         }
//         initControlPoints(control_points, false);
//         //initControlPointsForESDF(control_points, false); lambda2_ *= 2; //ESDF TEST
                             

//         printf("\033[32miter(+1)=%d,time=%5.3f,keep optimizing\n\033[0m", iter_num_, time_ms);
//       }
      

//       // cout << "[Optimization]: end-------------" << endl;
//     }
//     catch (std::exception& e)
//     {
//       t2 = clock();
//       double time_ms = (double)(t2-t1)/CLOCKS_PER_SEC*1000;
//       // printf("\033[36mfailure.iter=%d,time=%5.3f\n\033[0m", iter_num_, time_ms);
//       cout << e.what();
//       if ( flag_continue_to_optimize_ )
//       {
//         rebound_times ++;
//         cout << "(Doesn't matter) iter=" << iter_num_ << ",time=" << time_ms << endl;
//       }
//       else
//       {
//         cout << endl;
//         flag_nlopt_error_and_totally_fail = true;
//       }
      
//     }
//   } while ( (flag_continue_to_optimize_ || flag_occ ) && 
//             rebound_times <= 100 && 
//             restart_nums < max_restart_nums_set && 
//             !flag_nlopt_error_and_totally_fail && 
//             !success
//           );

//   lambda2_ = original_lambda2;

//   // if ( restart_nums < max_restart_nums_set || rebound_times > 100 ) return true; // rebound_times > 100? why???
//   return success;
// }


bool BsplineOptimizer::refine_optimize()
{
  /* ---------- initialize solver ---------- */
  iter_num_ = 0;
  int start_id = order_;
  int end_id = this->cps_.points.size() - order_;
  variable_num_ = 3 * (end_id - start_id);

  // cout << "variable_num_" << variable_num_ << endl;

  GradientDescentOptimizer opt(variable_num_, BsplineOptimizer::costFunctionRefine, this);

  opt.set_maxeval(100);
  opt.set_min_grad(1e-2);

  /* ---------- init variables ---------- */
  Eigen::VectorXd q(variable_num_);
  double final_cost;
  for (size_t i = start_id; i < end_id; ++i)
  {
    q.segment( 3 * (i - start_id), 3) = cps_.points[i];
  }

  double origin_lambda4 = lambda4_;
  bool flag_safe = true;
  int iter_count = 0;
  do
  {
    auto result = opt.optimize(q, final_cost);
    // cout << "result=" << result << endl;

    // cout << "cps_.points.size()=" << cps_.points.size() << " end_id=" << end_id << endl;
    // for (int i=0; i<cps_.points.size(); i++)
    //   cout << cps_.points[i].transpose() << endl;

    /* ---------- get results ---------- */
    Eigen::MatrixXd control_points(cps_.points.size(), 3);
    for ( int i=0; i<order_; i++ )
    {
      control_points.row(i) = cps_.points[i].transpose();
    }
    for (size_t i = start_id; i < end_id; ++i)
    {
      for (int j = 0; j < 3; j++)
        control_points(i,j) = best_variable_(3*(i-start_id) + j);
    }
    for ( int i=end_id; i<cps_.points.size(); i++ )
    {
      control_points.row(i) = cps_.points[i].transpose();
    }

    NonUniformBspline traj =  NonUniformBspline(control_points, 3, bspline_interval_);
    double tm, tmp;
    traj.getTimeSpan(tm, tmp);

    constexpr double t_step = 0.02;
    for ( double t = tm; t<tmp; t+=t_step )
    {
      if ( sdf_map_->getInflateOccupancy( traj.evaluateDeBoor(t) ) )
      {
        // cout << "Refined traj hit_obs, t=" << t << " P=" << traj.evaluateDeBoor(t).transpose() << endl;

        Eigen::MatrixXd ref_pts(ref_pts_.size(), 3);
        for ( int i=0; i<ref_pts_.size(); i++ )
        {
          ref_pts.row(i) = ref_pts_[i].transpose();
        }


        // NonUniformBspline ref_traj =  NonUniformBspline(ref_pts, 3, bspline_interval_);
        // double tm2, tmp2;
        // ref_traj.getTimeSpan(tm2, tmp2);
        // for ( double t2 = tm2; t2<tmp2; t2+=t_step )
        // {
        //   if ( sdf_map_->getInflateOccupancy( ref_traj.evaluateDeBoor(t2) ) )
        //   {
        //     ROS_WARN("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        //   }
        // }

        // cout << "ref_pts_=" << endl;
        // for ( auto e : ref_pts_ )
        // {
        //   cout << e.transpose() << endl;
        // }
        // cout << endl;
        // cout << "control_points=" << endl;
        // for ( int i=0; i<control_points.rows(); i++ )
        // {
        //   cout << control_points.row(i) << endl;
        // }

        flag_safe = false;
        break;
      }
    }



    // cout << "second_step=" << endl;
    // for ( double t = tm; t<tmp; t+=t_step )
    // {
    //   cout << traj.evaluateDeBoor(t).transpose() << endl;
    // }


    if ( !flag_safe ) lambda4_*=2;
    
    iter_count++;
  } while ( !flag_safe && iter_count <= 0 );

  // if ( iter_count > 1 && iter_count <=3 )
  // {
  //   ROS_ERROR("Refine iter_count > 1 && iter_count <=3");
  // }
  // if ( iter_count > 3 )
  // {
  //   cout << "iter_count=" << iter_count << endl;
  //   ROS_ERROR("Refine iter_count > 3");
  // }
  
  lambda4_ = origin_lambda4;

  //cout << "iter_num_=" << iter_num_ << endl;

  return flag_safe;
}

void BsplineOptimizer::combineCostRebound(const Eigen::VectorXd& x, Eigen::VectorXd& grad, double& f_combine)
{
  /* ---------- convert to control point vector ---------- */
  // vector<Eigen::Vector3d> q;
  // q.reserve( cps_.size );

  // /* first p points */
  // for (int i = 0; i < order_; i++)
  //   q.push_back(cps_.points[i]);

  /* optimized control points */
  for (int i = 0; i < variable_num_ / 3; i++)
  {
    cps_.points[i+order_] = x.segment(3*i, 3);
  }

  // /* last p points */
  // for (int i = 0; i < order_; i++)
  //   q.push_back(cps_.points[cps_.size - order_ + i]);

  for ( size_t i=order_; i<cps_.size-order_; ++i )
  {
    cps_.occupancy[i] = sdf_map_->getInflateOccupancy(cps_.points[i]);
  }

  // for ( int i=0; i<cps_.size(); i++ )
  //   cout << cps_[i].point.transpose() << endl;
  // cout << endl;

  /* ---------- evaluate cost and gradient ---------- */
  double f_smoothness, f_distance, f_feasibility;

  vector<Eigen::Vector3d> g_smoothness, g_distance, g_feasibility;
  g_smoothness.resize(cps_.size);
  g_distance.resize(cps_.size);
  g_feasibility.resize(cps_.size);

  //time_satrt = ros::Time::now();

  // for ( int i=0; i<cps_.points.size(); i++ )
  //   cout << cps_.points[i].transpose() << endl;

  calcSmoothnessCost(cps_.points, f_smoothness, g_smoothness);
  calcDistanceCostRebound(cps_.points, f_distance, g_distance, iter_num_, f_smoothness);
  calcFeasibilityCost(cps_.points, f_feasibility, g_feasibility);

  // for ( auto e : g_smoothness )
  //   cout << e.transpose() << endl;
  // for ( auto e : g_distance )
  //   cout << e.transpose() << endl;
  // for ( auto e : g_feasibility )
  //   cout << e.transpose() << endl;

  f_combine = lambda1_ * f_smoothness + lambda2_ * f_distance + lambda3_ * f_feasibility;
  //printf("origin %f %f %f %f\n", f_smoothness, f_distance, f_feasibility, f_combine);


  grad.resize(variable_num_);
  for (int i = 0; i < variable_num_ / 3; i++)
  {
    
    for (int j = 0; j < 3; j++)
    {
      /* the first p points is static here */
      grad(3 * i + j) = lambda1_ * g_smoothness[i + order_](j) + lambda2_ * g_distance[i + order_](j) +
                        lambda3_ * g_feasibility[i + order_](j);

    }
    //cout << "g_smoothness=" << g_smoothness[i + order_].transpose() << " g_distance=" << g_distance[i + order_].transpose() << " g_feasibility=" << g_feasibility[i + order_].transpose() << endl;
  }

  // cout << grad.transpose() << endl;
}

void BsplineOptimizer::combineCostRefine(const Eigen::VectorXd& x, Eigen::VectorXd& grad, double& f_combine)
{
  /* ---------- convert to control point vector ---------- */
  //vector<Eigen::Vector3d> q(cps_.points.size());

  // /* first p points */
  // for (int i = 0; i < order_; i++)
  //   q[i] = control_points_.row(i).transpose();

  /* optimized control points */
  for (int i = 0; i < variable_num_ / 3; i++)
  {
    Eigen::Vector3d qi(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    cps_.points[i+order_] = qi;
  }

  // /* last p points */
  // for (int i = 0; i < order_; i++)
  //   q.push_back(control_points_.row(control_points_.rows()-order_+i));

  /* ---------- evaluate cost and gradient ---------- */
  double f_smoothness, f_fitness, f_feasibility;

  vector<Eigen::Vector3d> g_smoothness, g_fitness, g_feasibility;
  g_smoothness.resize(cps_.points.size());
  g_fitness.resize(cps_.points.size());
  g_feasibility.resize(cps_.points.size());

  //time_satrt = ros::Time::now();

  calcSmoothnessCost(cps_.points, f_smoothness, g_smoothness);
  calcFitnessCost(cps_.points, f_fitness, g_fitness);
  calcFeasibilityCost(cps_.points, f_feasibility, g_feasibility);

  /* ---------- convert to NLopt format...---------- */
  f_combine = lambda1_ * f_smoothness + lambda4_ * f_fitness + lambda3_ * f_feasibility;
  // printf("origin %f %f %f %f\n", f_smoothness, f_fitness, f_feasibility, f_combine);

  grad.resize(variable_num_);
  for (int i = 0; i < variable_num_ / 3; i++)
  {
    
    for (int j = 0; j < 3; j++)
    {
      /* the first p points is static here */
      grad[3 * i + j] = lambda1_ * g_smoothness[i + order_](j) + lambda4_ * g_fitness[i + order_](j) +
                        lambda3_ * g_feasibility[i + order_](j);

    }
    //cout << "g_smoothness=" << g_smoothness[i + order_].transpose() << " g_distance=" << g_distance[i + order_].transpose() << " g_feasibility=" << g_feasibility[i + order_].transpose() << endl;
  }
}


}  // namespace rebound_planner