// Generated by gencpp from file hybrid_simulation/ChangeLane.msg
// DO NOT EDIT!


#ifndef HYBRID_SIMULATION_MESSAGE_CHANGELANE_H
#define HYBRID_SIMULATION_MESSAGE_CHANGELANE_H


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
struct ChangeLane_
{
  typedef ChangeLane_<ContainerAllocator> Type;

  ChangeLane_()
    : lane_change(0)  {
    }
  ChangeLane_(const ContainerAllocator& _alloc)
    : lane_change(0)  {
  (void)_alloc;
    }



   typedef int16_t _lane_change_type;
  _lane_change_type lane_change;





  typedef boost::shared_ptr< ::hybrid_simulation::ChangeLane_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::hybrid_simulation::ChangeLane_<ContainerAllocator> const> ConstPtr;

}; // struct ChangeLane_

typedef ::hybrid_simulation::ChangeLane_<std::allocator<void> > ChangeLane;

typedef boost::shared_ptr< ::hybrid_simulation::ChangeLane > ChangeLanePtr;
typedef boost::shared_ptr< ::hybrid_simulation::ChangeLane const> ChangeLaneConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::hybrid_simulation::ChangeLane_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::hybrid_simulation::ChangeLane_<ContainerAllocator1> & lhs, const ::hybrid_simulation::ChangeLane_<ContainerAllocator2> & rhs)
{
  return lhs.lane_change == rhs.lane_change;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::hybrid_simulation::ChangeLane_<ContainerAllocator1> & lhs, const ::hybrid_simulation::ChangeLane_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace hybrid_simulation

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::ChangeLane_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::ChangeLane_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::ChangeLane_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
{
  static const char* value()
  {
    return "21070bac28cd495dd1acc43133eea981";
  }

  static const char* value(const ::hybrid_simulation::ChangeLane_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x21070bac28cd495dULL;
  static const uint64_t static_value2 = 0xd1acc43133eea981ULL;
};

template<class ContainerAllocator>
struct DataType< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
{
  static const char* value()
  {
    return "hybrid_simulation/ChangeLane";
  }

  static const char* value(const ::hybrid_simulation::ChangeLane_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Message to control high level actions of the Ego-Vehicle\n"
"\n"
"\n"
"# lane_change : Change lane (0 keep lane; 1 lane change right; 2 lane change left)\n"
"\n"
"int16 lane_change\n"
;
  }

  static const char* value(const ::hybrid_simulation::ChangeLane_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.lane_change);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ChangeLane_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::hybrid_simulation::ChangeLane_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::hybrid_simulation::ChangeLane_<ContainerAllocator>& v)
  {
    s << indent << "lane_change: ";
    Printer<int16_t>::stream(s, indent + "  ", v.lane_change);
  }
};

} // namespace message_operations
} // namespace ros

#endif // HYBRID_SIMULATION_MESSAGE_CHANGELANE_H
