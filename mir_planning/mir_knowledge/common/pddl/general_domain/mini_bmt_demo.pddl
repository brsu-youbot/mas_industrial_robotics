;This PDDL problem definition was made automatically from a KB snapshot
(define (problem general_domain_task)
    (:domain general_domain)

    (:objects
        WS01 WS02 WS03 WS04 WS05 WS06 SH01 TT01 PP01 START EXIT - location
        ALLENKEY-00 AXIS2-00 BEARING2-00 DRILL-00 F20_20_B-00 F20_20_G-00 F20_20_G-01 HOUSING-00 M20-00 M20_100-00 M30-00 M30-01 MOTOR2-00 S40_40_B-00 S40_40_G-00 SCREWDRIVER-00 SPACER-00 WRENCH-00 CONTAINER_RED-00 CONTAINER_BLUE-00 PP01_CAVITY-00 - object
        YOUBOT-BRSU - robot
        PLATFORM_LEFT PLATFORM_MIDDLE PLATFORM_RIGHT - robot_platform
    )

    (:init
        ;Cost information starts
        (= (total-cost) 0)
        ;Cost information ends

        (at YOUBOT-BRSU START)
        (gripper_is_free YOUBOT-BRSU)
        (on MOTOR2-00 SH01)

        (on S40_40_G-00 WS01)
        (on M30-00 WS01)

        (on BEARING2-00 WS02)

        (on F20_20_B-00 WS03)

        (on S40_40_B-00 TT01)

        (container PP01_CAVITY-00)
        (heavy PP01_CAVITY-00)
        (on PP01_CAVITY-00 PP01)
        
        (is_large ALLENKEY-00)
        (is_big_enough PLATFORM_RIGHT)
    )

    (:goal
        (and
            (in S40_40_G-00 PP01_CAVITY-00)
            (on MOTOR2-00 WS01)
            (on M30-00 WS02)
            (on BEARING2-00 WS03)
            (on F20_20_B-00 SH01)
            (on S40_40_B-00 WS04)
        )
    )

    (:metric minimize
        (total-cost)
    )

)
