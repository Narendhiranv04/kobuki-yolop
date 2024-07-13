// Generated by gencpp from file hybrid_simulation/VehicleStatus.msg
// DO NOT EDIT!


#ifndef HYBRID_SIMULATION_MESSAGE_VEHICLESTATUS_H
#define HYBRID_SIMULATION_MESSAGE_VEHICLESTATUS_H


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
struct VehicleStatus_
{
  typedef VehicleStatus_<ContainerAllocator> Type;

  VehicleStatus_()
    : vehicle_id()
    , pos_x(0.0)
    , pos_y(0.0)
    , heading(0.0)
    , velocity(0.0)
    , max_vel(0.0)
    , lane(0)
    , signals(0)  {
    }
  VehicleStatus_(const ContainerAllocator& _alloc)
    : vehicle_id(_alloc)
    , pos_x(0.0)
    , pos_y(0.0)
    , heading(0.0)
    , velocity(0.0)
    , max_vel(0.0)
    , lane(0)
    , signals(0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _vehicle_id_type;
  _vehicle_id_type vehicle_id;

   typedef float _pos_x_type;
  _pos_x_type pos_x;

   typedef float _pos_y_type;
  _pos_y_type pos_y;

   typedef float _heading_type;
  _heading_type heading;

   typedef float _velocity_type;
  _velocity_type velocity;

   typedef float _max_vel_type;
  _max_vel_type max_vel;

   typedef int16_t _lane_type;
  _lane_type lane;

   typedef int16_t _signals_type;
  _signals_type signals;





  typedef boost::shared_ptr< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> const> ConstPtr;

}; // struct VehicleStatus_

typedef ::hybrid_simulation::VehicleStatus_<std::allocator<void> > VehicleStatus;

typedef boost::shared_ptr< ::hybrid_simulation::VehicleStatus > VehicleStatusPtr;
typedef boost::shared_ptr< ::hybrid_simulation::VehicleStatus const> VehicleStatusConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::hybrid_simulation::VehicleStatus_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::hybrid_simulation::VehicleStatus_<ContainerAllocator1> & lhs, const ::hybrid_simulation::VehicleStatus_<ContainerAllocator2> & rhs)
{
  return lhs.vehicle_id == rhs.vehicle_id &&
    lhs.pos_x == rhs.pos_x &&
    lhs.pos_y == rhs.pos_y &&
    lhs.heading == rhs.heading &&
    lhs.velocity == rhs.velocity &&
    lhs.max_vel == rhs.max_vel &&
    lhs.lane == rhs.lane &&
    lhs.signals == rhs.signals;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::hybrid_simulation::VehicleStatus_<ContainerAllocator1> & lhs, const ::hybrid_simulation::VehicleStatus_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace hybrid_simulation

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "c81aa0791049124d486b5aa675fa06f6";
  }

  static const char* value(const ::hybrid_simulation::VehicleStatus_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xc81aa0791049124dULL;
  static const uint64_t static_value2 = 0x486b5aa675fa06f6ULL;
};

template<class ContainerAllocator>
struct DataType< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "hybrid_simulation/VehicleStatus";
  }

  static const char* value(const ::hybrid_simulation::VehicleStatus_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Message to send information about Vehicles in the scene\n"
"\n"
"# id : The idenfification of the vehicle\n"
"# pos_x : Vehicle x position\n"
"# pos_y : Vehicle y position\n"
"# heading : Vehicle heading (Yaw angle)\n"
"# velocity : Linear velocity of the vehicle\n"
"# max_vel : Maximum velocity\n"
"\n"
"string  vehicle_id\n"
"float32 pos_x\n"
"float32 pos_y\n"
"float32 heading\n"
"float32 velocity\n"
"float32 max_vel\n"
"int16 lane\n"
"int16 signals\n"
"\n"
;
  }

  static const char* value(const ::hybrid_simulation::VehicleStatus_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.vehicle_id);
      stream.next(m.pos_x);
      stream.next(m.pos_y);
      stream.next(m.heading);
      stream.next(m.velocity);
      stream.next(m.max_vel);
      stream.next(m.lane);
      stream.next(m.signals);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct VehicleStatus_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::hybrid_simulation::VehicleStatus_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::hybrid_simulation::VehicleStatus_<ContainerAllocator>& v)
  {
    s << indent << "vehicle_id: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.vehicle_id);
    s << indent << "pos_x: ";
    Printer<float>::stream(s, indent + "  ", v.pos_x);
    s << indent << "pos_y: ";
    Printer<float>::stream(s, indent + "  ", v.pos_y);
    s << indent << "heading: ";
    Printer<float>::stream(s, indent + "  ", v.heading);
    s << indent << "velocity: ";
    Printer<float>::stream(s, indent + "  ", v.velocity);
    s << indent << "max_vel: ";
    Printer<float>::stream(s, indent + "  ", v.max_vel);
    s << indent << "lane: ";
    Printer<int16_t>::stream(s, indent + "  ", v.lane);
    s << indent << "signals: ";
    Printer<int16_t>::stream(s, indent + "  ", v.signals);
  }
};

} // namespace message_operations
} // namespace ros

#endif // HYBRID_SIMULATION_MESSAGE_VEHICLESTATUS_H
