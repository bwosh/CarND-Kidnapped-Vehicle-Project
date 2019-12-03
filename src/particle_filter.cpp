/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

#define PARTICLES_NUMBER 100
#define THETA_EPS 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = PARTICLES_NUMBER;

  // Normal distribtions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{

  // normal distributions 0-centered
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // New state calculation
  for (int i = 0; i < num_particles; ++i)
  {
    Particle *p = &(particles[i]);

    double theta = p->theta;

    if (fabs(yaw_rate) < THETA_EPS)
    {
      p->x += velocity * delta_t * cos(theta);
      p->y += velocity * delta_t * sin(theta);
      // yaw stays the same
    }
    else
    {
      p->x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      p->y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      p->theta += yaw_rate * delta_t;
    }

    // Add noise
    p->x += dist_x(gen);
    p->y += dist_y(gen);
    p->theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  int o_size = observations.size();
  int p_size = predicted.size();

  // Iterate through observations
  for (unsigned int o = 0; o < o_size; o++)
  {

    LandmarkObs *observation = &(observations[o]);

    double min_distance = numeric_limits<double>::max();
    int mapId = -1;

    // Iterate through predictions
    for (unsigned p = 0; p < p_size; p++)
    {
      LandmarkObs *prediction = &(predicted[p]);

      double x_dist = observation->x - prediction->x;
      double y_dist = observation->y - prediction->y;
      double distance = x_dist * x_dist + y_dist * y_dist;

      if (distance < min_distance)
      {
        min_distance = distance;
        mapId = prediction->id;
      }
    }

    // Save closest point id
    observation->id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  double std_range = std_landmark[0];
  double std_bearing = std_landmark[1];

  for (int i = 0; i < num_particles; ++i) 
  {
    Particle *p = &(particles[i]);

    // Filter landmarks to range of sensor
    double sensor_range_squared = sensor_range * sensor_range;
    vector<LandmarkObs> inrange_landmarks;
    for(int l = 0; l < map_landmarks.landmark_list.size(); ++l) 
    {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[l];
      float l_x = landmark.x_f;
      float l_y = landmark.y_f;

      double dX = p->x - l_x;
      double dY = p->y - l_y;
      if ( dX*dX + dY*dY <= sensor_range_squared ) 
      {
        inrange_landmarks.push_back(LandmarkObs{ landmark.id_i, l_x, l_y });
      }
    }

    // Change coordinates to transformed
    vector<LandmarkObs> mapped_observations;
    for(unsigned int j = 0; j < observations.size(); j++) 
    {
      double mapped_x = cos(p->theta)*observations[j].x - sin(p->theta)*observations[j].y + p->x;
      double mapped_y = sin(p->theta)*observations[j].x + cos(p->theta)*observations[j].y + p->y;
      mapped_observations.push_back(LandmarkObs{ observations[j].id, mapped_x, mapped_y });
    }
  }


}

void ParticleFilter::resample()
{
  // Get info about weights
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();

  for(int i = 0; i < num_particles; ++i) 
  {
    Particle *particle = &(particles[i]);
    // save weight
    weights.push_back(particle->weight);

    // save maximum weight
    if ( particle->weight > max_weight ) 
    {
      max_weight = particle->weight;
    }
  }

  // initializing  wheel method
  vector<Particle> resampled;
  int index = uniform_int_distribution<int>(0, num_particles - 1)(gen);
  double beta = 0.0;
  uniform_real_distribution<double> distDouble(0.0, max_weight);

  for(int i = 0; i < num_particles; ++i) 
  {
    beta += distDouble(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }

    resampled.push_back(particles[index]);
  }

  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}