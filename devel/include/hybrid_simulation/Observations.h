// Generated by gencpp from file hybrid_simulation/Observations.msg
// DO NOT EDIT!


#ifndef HYBRID_SIMULATION_MESSAGE_OBSERVATIONS_H
#define HYBRID_SIMULATION_MESSAGE_OBSERVATIONS_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace hybrid_simulation
{
template <class ContainerAllocator>
struct Observations_
{
  typedef Observations_<ContainerAllocator> Type;

  Observations_()
    : front_left(0)
    , front(0)
    , front_right(0)
    , center_left(0)
    , center_right(0)
    , rear_left(0)
    , rear_right(0)
    , back_left(0)
    , back_right(0)
    , lane(0)
    , dist_goal(0.0)  {
    }
  Observations_(const ContainerAllocator& _alloc)
    : front_left(0)
    , front(0)
    , front_right(0)
    , center_left(0)
    , center_right(0)
    , rear_left(0)
    , rear_right(0)
    , back_left(0)
    , back_right(0)
    , lane(0)
    , dist_goal(0.0)  {
  (void)_alloc;
    }



   typedef int8_t _front_left_type;
  _front_left_type front_left;

   typedef int8_t _front_type;
  _front_type front;

   typedef int8_t _front_right_type;
  _front_right_type front_right;

   typedef int8_t _center_left_type;
  _center_left_type center_left;

   typedef int8_t _center_right_type;
  _center_right_type center_right;

   typedef int8_t _rear_left_type;
  _rear_left_type rear_left;

   typedef int8_t _rear_right_type;
  _rear_right_type rear_right;

   typedef int8_t _back_left_type;
  _back_left_type back_left;

   typedef int8_t _back_right_type;
  _back_right_type back_right;

   typedef int8_t _lane_type;
  _lane_type lane;

   typedef float _dist_goal_type;
  _dist_goal_type dist_goal;





  typedef boost::shared_ptr< ::hybrid_simulation::Observations_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::hybrid_simulation::Observations_<ContainerAllocator> const> ConstPtr;

}; // struct Observations_

typedef ::hybrid_simulation::Observations_<std::allocator<void> > Observations;

typedef boost::shared_ptr< ::hybrid_simulation::Observations > ObservationsPtr;
typedef boost::shared_ptr< ::hybrid_simulation::Observations const> ObservationsConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::hybrid_simulation::Observations_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::hybrid_simulation::Observations_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::hybrid_simulation::Observations_<ContainerAllocator1> & lhs, const ::hybrid_simulation::Observations_<ContainerAllocator2> & rhs)
{
  return lhs.front_left == rhs.front_left &&
    lhs.front == rhs.front &&
    lhs.front_right == rhs.front_right &&
    lhs.center_left == rhs.center_left &&
    lhs.center_right == rhs.center_right &&
    lhs.rear_left == rhs.rear_left &&
    lhs.rear_right == rhs.rear_right &&
    lhs.back_left == rhs.back_left &&
    lhs.back_right == rhs.back_right &&
    lhs.lane == rhs.lane &&
    lhs.dist_goal == rhs.dist_goal;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::hybrid_simulation::Observations_<ContainerAllocator1> & lhs, const ::hybrid_simulation::Observations_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace hybrid_simulation

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::Observations_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::Observations_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::Observations_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::Observations_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::Observations_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::Observations_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::hybrid_simulation::Observations_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a9c83c991797fc3e633dc6b433db3a15";
  }

  static const char* value(const ::hybrid_simulation::Observations_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa9c83c991797fc3eULL;
  static const uint64_t static_value2 = 0x633dc6b433db3a15ULL;
};

template<class ContainerAllocator>
struct DataType< ::hybrid_simulation::Observations_<ContainerAllocator> >
{
  static const char* value()
  {
    return "hybrid_simulation/Observations";
  }

  static const char* value(const ::hybrid_simulation::Observations_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::hybrid_simulation::Observations_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Message of the observations for decision making\n"
"\n"
"# Possible_speeds: -100 Free;  0 Static; 1 Slow; 2 Fast; 100 Blocked\n"
"\n"
"int8 front_left\n"
"int8 front\n"
"int8 front_right\n"
"int8 center_left\n"
"int8 center_right\n"
"int8 rear_left\n"
"int8 rear_right\n"
"int8 back_left\n"
"int8 back_right\n"
"# lane: -1 right of goal;  0 goal lane; 1 Left of lane\n"
"int8 lane\n"
"# dist_goal: Distance (m) to end of road / exit / end lane\n"
"float32 dist_goal\n"
;
  }

  static const char* value(const ::hybrid_simulation::Observations_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::hybrid_simulation::Observations_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.front_left);
      stream.next(m.front);
      stream.next(m.front_right);
      stream.next(m.center_left);
      stream.next(m.center_right);
      stream.next(m.rear_left);
      stream.next(m.rear_right);
      stream.next(m.back_left);
      stream.next(m.back_right);
      stream.next(m.lane);
      stream.next(m.dist_goal);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Observations_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::hybrid_simulation::Observations_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::hybrid_simulation::Observations_<ContainerAllocator>& v)
  {
    s << indent << "front_left: ";
    Printer<int8_t>::stream(s, indent + "  ", v.front_left);
    s << indent << "front: ";
    Printer<int8_t>::stream(s, indent + "  ", v.front);
    s << indent << "front_right: ";
    Printer<int8_t>::stream(s, indent + "  ", v.front_right);
    s << indent << "center_left: ";
    Printer<int8_t>::stream(s, indent + "  ", v.center_left);
    s << indent << "center_right: ";
    Printer<int8_t>::stream(s, indent + "  ", v.center_right);
    s << indent << "rear_left: ";
    Printer<int8_t>::stream(s, indent + "  ", v.rear_left);
    s << indent << "rear_right: ";
    Printer<int8_t>::stream(s, indent + "  ", v.rear_right);
    s << indent << "back_left: ";
    Printer<int8_t>::stream(s, indent + "  ", v.back_left);
    s << indent << "back_right: ";
    Printer<int8_t>::stream(s, indent + "  ", v.back_right);
    s << indent << "lane: ";
    Printer<int8_t>::stream(s, indent + "  ", v.lane);
    s << indent << "dist_goal: ";
    Printer<float>::stream(s, indent + "  ", v.dist_goal);
  }
};

} // namespace message_operations
} // namespace ros

#endif // HYBRID_SIMULATION_MESSAGE_OBSERVATIONS_H
